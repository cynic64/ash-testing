use ash::extensions::khr::{Swapchain, XlibSurface};
use ash::extensions::{ext::DebugUtils, khr::Surface};
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk::DeviceSize;
use ash::{vk, Device, Instance};

use crossbeam_channel::{Receiver, Sender};

use std::ptr;
use winit::{dpi::LogicalPosition, ElementState, Event, EventsLoop, MouseButton, WindowEvent};

use crate::{get_elapsed, size_of_slice, LoopTimer};

pub struct Renderer<V> {
    device: Device,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    swapchain_creator: Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_dims: vk::Extent2D,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    must_recreate_swapchain: bool,
    surface_loader: Surface,
    surface: vk::SurfaceKHR,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    shader_stages: [vk::PipelineShaderStageCreateInfo; 2],
    binding_descriptions: [vk::VertexInputBindingDescription; 1],
    attribute_descriptions: [vk::VertexInputAttributeDescription; 2],
    command_pool: vk::CommandPool,
    queue_family_index: u32,
    flying_frames: Vec<FlyingFrame>,
    vertex_buffer_capacity: DeviceSize,
    index_buffer_capacity: DeviceSize,

    events_loop: EventsLoop,
    mesh_recv: Receiver<Mesh<V>>,
    event_send: Sender<WindowEvent>,

    swapchain_fences: Vec<Option<vk::Fence>>,
    frames_drawn: usize,
    start_time: std::time::Instant,
    mouse_pos: PixelPos,
    mouse_down: bool,
    last_sent_mouse_pos: PixelPos,

    timer_mesh_wait: LoopTimer,
    timer_draw: LoopTimer,
    timer_write_vbuf: LoopTimer,
    timer_copy_vbuf: LoopTimer,
    timer_write_ibuf: LoopTimer,
    timer_copy_ibuf: LoopTimer,
    timer_copy_complete: LoopTimer,
    timer_create_cbuf: LoopTimer,
}

pub struct Mesh<V> {
    pub vertices: Vec<V>,
    pub indices: Vec<IndexType>,
}

// range from 0.0 .. screen size
type PixelPos = [f64; 2];
// -1.0 .. 1.0
type VkPos = [f64; 2];
type IndexType = u32;

const MAX_FRAMES_IN_FLIGHT: usize = 4;
const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;
const MAX_POINT_DIST: f64 = 10.0;

// we cannot know which image will be used before calling acquire_next_image,
// for which we already need a semaphore ready for it to signal.

// therefore, there is no direct link between FlyingFrames and swapchain images
// - each time we draw a new frame, wait for the oldest sync_set to finish and
// use that.
struct FlyingFrame {
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    render_finished_fence: vk::Fence,
    command_buffer: Option<vk::CommandBuffer>,

    vertex_staging_buffer: vk::Buffer,
    vertex_staging_buffer_memory: vk::DeviceMemory,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_staging_buffer: vk::Buffer,
    index_staging_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    vbuf_copied_fence: vk::Fence,
    ibuf_copied_fence: vk::Fence,
}

impl<V> Renderer<V> {
    pub fn new(
        device: Device,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        queue: vk::Queue,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        swapchain_creator: Swapchain,
        swapchain: vk::SwapchainKHR,
        swapchain_dims: vk::Extent2D,
        swapchain_images: Vec<vk::Image>,
        swapchain_image_views: Vec<vk::ImageView>,
        surface_loader: Surface,
        surface: vk::SurfaceKHR,
        render_pass: vk::RenderPass,
        shader_stages: [vk::PipelineShaderStageCreateInfo; 2],
        binding_descriptions: [vk::VertexInputBindingDescription; 1],
        attribute_descriptions: [vk::VertexInputAttributeDescription; 2],
        vertex_buffer_capacity: u64,
        index_buffer_capacity: u64,
        queue_family_index: u32,
        events_loop: EventsLoop,
        mesh_recv: Receiver<Mesh<V>>,
        event_send: Sender<WindowEvent>,
    ) -> Self {
        println!("Maximum frames in flight: {} ", MAX_FRAMES_IN_FLIGHT);

        // framebuffer creation
        let framebuffers =
            create_framebuffers(&device, render_pass, swapchain_dims, &swapchain_image_views);

        // flying frames (one for each swapchain image)
        let device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let flying_frames = setup_flying_frames(
            &device,
            device_memory_properties,
            vertex_buffer_capacity,
            index_buffer_capacity,
        );

        // command pool
        let command_pool = create_command_pool(&device, queue_family_index);

        // timers
        let timer_mesh_wait = LoopTimer::new("Waiting on mesh gen".to_string());
        let timer_draw = LoopTimer::new("Drawing".to_string());
        let timer_write_vbuf = LoopTimer::new("Write vbuf".to_string());
        let timer_copy_vbuf = LoopTimer::new("Copy vbuf".to_string());
        let timer_write_ibuf = LoopTimer::new("Write ibuf".to_string());
        let timer_copy_ibuf = LoopTimer::new("Copy ibuf".to_string());
        let timer_copy_complete = LoopTimer::new("Copy completion".to_string());
        let timer_create_cbuf = LoopTimer::new("Create command buffer".to_string());

        let swapchain_image_count = swapchain_images.len();

        Self {
            device: device.clone(),
            physical_device,
            queue,
            must_recreate_swapchain: false,
            framebuffers,
            swapchain_dims,
            swapchain_images,
            swapchain_image_views,
            swapchain_creator,
            pipeline,
            surface,
            render_pass,
            pipeline_layout,
            shader_stages,
            binding_descriptions,
            surface_loader,
            attribute_descriptions,
            swapchain,
            command_pool,
            queue_family_index,
            flying_frames,
            vertex_buffer_capacity,
            index_buffer_capacity,

            events_loop,

            mesh_recv,
            event_send,

            swapchain_fences: vec![None; swapchain_image_count],
            frames_drawn: 0,
            start_time: std::time::Instant::now(),
            mouse_pos: [-9999.0, -9999.0],
            mouse_down: false,
            last_sent_mouse_pos: [-9999.0, -9999.0],

            timer_mesh_wait,
            timer_draw,
            timer_write_vbuf,
            timer_copy_vbuf,
            timer_write_ibuf,
            timer_copy_ibuf,
            timer_copy_complete,
            timer_create_cbuf,
        }
    }

    pub fn start(&mut self) {
        loop {
            if self.must_recreate_swapchain {
                unsafe { self.device.device_wait_idle() }
                    .expect("Couldn't wait for self.device to become idle");

                // Cleanup_swapchain requires ownership of these to destroy
                // them. They will be re-created later. Technically I think it
                // would also work to pass pointers to cleanup_swapchain instead
                // of transferring ownership, but I don't like the idea of that
                // because then values in command_buffers and framebuffers would
                // point to destroyed vulkan objects.
                let our_framebuffers = std::mem::replace(&mut self.framebuffers, vec![]);
                let our_swapchain_image_views =
                    std::mem::replace(&mut self.swapchain_image_views, vec![]);

                cleanup_swapchain(
                    &self.device,
                    &self.swapchain_creator,
                    our_framebuffers,
                    our_swapchain_image_views,
                    self.swapchain,
                );

                // re-create swapchain
                let ret = create_swapchain(
                    self.physical_device,
                    &self.surface_loader,
                    self.surface,
                    &self.swapchain_creator,
                );

                self.swapchain = ret.0;
                self.swapchain_images = ret.1;
                self.swapchain_dims = ret.2;

                // re-create image views
                self.swapchain_image_views = create_swapchain_image_views(&self.device, &self.swapchain_images);

                // re-create framebuffers
                // framebuffer creation
                self.framebuffers = create_framebuffers(
                    &self.device,
                    self.render_pass,
                    self.swapchain_dims,
                    &self.swapchain_image_views,
                );

                self.must_recreate_swapchain = false;
            }

            let mut exit = false;

            // necessary because the borrow checker can't figure out that
            // self.events_loop.poll_events() doesn't borrow all of self
            let event_send = &self.event_send;
            self.events_loop.poll_events(|ev| match ev {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => exit = true,
                Event::WindowEvent { event, .. } => event_send.send(event).unwrap(),
                _ => {}
            });

            if exit {
                break;
            }

            let flying_frame_idx = self.frames_drawn % MAX_FRAMES_IN_FLIGHT;
            let frame = &self.flying_frames[flying_frame_idx];

            let image_available_semaphore = frame.image_available_semaphore;
            let render_finished_semaphore = frame.render_finished_semaphore;
            let render_finished_fence = frame.render_finished_fence;

            // we can't use this sync set until whichever rendering was using it
            // previously is finished, so wait for rendering to finished (use the
            // fence because it's a GPU - CPU sync)
            unsafe {
                self.device
                    .wait_for_fences(&[render_finished_fence], true, std::u64::MAX)
            }
            .expect("Couldn't wait for previous sync set to finish rendering");

            // since rendering has finished, we can free that command buffer
            if let Some(command_buffer) = frame.command_buffer {
                unsafe {
                    self.device
                        .free_command_buffers(self.command_pool, &[command_buffer])
                };
            } else {
                // should only happen when swapchain images are first used, panic
                // otherwise
                if self.frames_drawn >= MAX_FRAMES_IN_FLIGHT {
                    panic!(
                        "No command buffer in flying frame on frame {}",
                        self.frames_drawn
                    );
                }
            }

            // image_available_semaphore will be signalled once the swapchain image
            // is actually available and not being displayed anymore -
            // acquire_next_image will return the instant it knows which image index
            // will be free next, so we need to wait on that semaphore
            let acquire_result = unsafe {
                self.swapchain_creator.acquire_next_image(
                    self.swapchain,
                    std::u64::MAX,
                    image_available_semaphore,
                    vk::Fence::null(),
                )
            };

            let image_idx = match acquire_result {
                Ok((image_idx, _is_sub_optimal)) => image_idx,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.must_recreate_swapchain = true;
                    continue;
                }
                Err(e) => panic!("Unexpected error during acquire_next_image: {}", e),
            };

            // because we might have more sync sets than swapchain images, it's
            // possible another set of sync sets is still rendering to this
            // swapchain image. by waiting on the fence associated with this
            // swapchain image, we can ensure it really is available
            if let Some(image_fence) = self.swapchain_fences[image_idx as usize] {
                unsafe {
                    self.device
                        .wait_for_fences(&[image_fence], true, std::u64::MAX)
                }
                .expect("Couldn't wait for image_in_flight fence");
            } else {
                // this should only happen on the first MAX_FRAMES_IN_FLIGHT frames
                // drawn, because at that point we will not yet have started using
                // all sync sets

                // if it happens afterwards, panic
                if self.frames_drawn >= MAX_FRAMES_IN_FLIGHT {
                    panic!(
                        "No fence for this swapchain image at frame {}! image_idx: {}",
                        self.frames_drawn, image_idx
                    );
                }
            }

            self.timer_draw.stop();

            self.timer_mesh_wait.start();
            let mesh = self.mesh_recv.recv().expect("Error when receiving mesh");
            self.timer_mesh_wait.stop();

            // set the render_finished_fence associated with this swapchain image to
            // the fence that will be signalled when we finish rendering - in other
            // words, "We're using this image! Don't touch it till we finish."
            self.swapchain_fences[image_idx as usize] = Some(render_finished_fence);

            // change mesh data on GPU

            // make sure we won't go past the memory we've allocated
            let vertex_buffer_size = size_of_slice(&mesh.vertices);
            let index_buffer_size = size_of_slice(&mesh.indices);

            assert!(
                vertex_buffer_size < self.vertex_buffer_capacity,
                "Vertex buffers ({}) too small for vertex data ({})!",
                self.vertex_buffer_capacity,
                vertex_buffer_size
            );

            assert!(
                index_buffer_size < self.index_buffer_capacity,
                "Index buffers ({}) too small for index data ({})!",
                self.index_buffer_capacity,
                index_buffer_size
            );

            // even though we modify these, we don't need to mutably borrow because
            // they are magical Vulkan pointers that don't care about lifetimes
            let vertex_staging_buffer = self.flying_frames[flying_frame_idx].vertex_staging_buffer;
            let vertex_staging_buffer_memory =
                self.flying_frames[flying_frame_idx].vertex_staging_buffer_memory;
            let vertex_buffer = self.flying_frames[flying_frame_idx].vertex_buffer;
            let index_buffer = self.flying_frames[flying_frame_idx].index_buffer;
            let index_staging_buffer = self.flying_frames[flying_frame_idx].index_staging_buffer;
            let index_staging_buffer_memory =
                self.flying_frames[flying_frame_idx].index_staging_buffer_memory;

            let vbuf_copied_fence = self.flying_frames[flying_frame_idx].vbuf_copied_fence;
            let ibuf_copied_fence = self.flying_frames[flying_frame_idx].ibuf_copied_fence;

            // map and write vertex data to vertex staging buffer
            let mut copy_command_buffers = None;
            if mesh.vertices.len() > 0 {
                // reset the fences that will be used
                unsafe {
                    self.device
                        .reset_fences(&[vbuf_copied_fence, ibuf_copied_fence])
                }
                .expect("Couldn't reset vbuf_ and ibuf_copied_fence");

                self.timer_write_vbuf.start();
                write_to_cpu_accessible_buffer(
                    &self.device,
                    vertex_staging_buffer_memory,
                    &mesh.vertices,
                );
                self.timer_write_vbuf.stop();

                // copy staging buffer to vertex buffer
                self.timer_copy_vbuf.start();
                let vbuf_command_buffer = copy_buffer(
                    &self.device,
                    self.queue,
                    self.command_pool,
                    vertex_staging_buffer,
                    vertex_buffer,
                    vertex_buffer_size,
                    vbuf_copied_fence,
                );
                self.timer_copy_vbuf.stop();

                // map and write index data to staging buffer
                self.timer_write_ibuf.start();
                write_to_cpu_accessible_buffer(
                    &self.device,
                    index_staging_buffer_memory,
                    &mesh.indices,
                );
                self.timer_write_ibuf.stop();

                // copy staging buffer to index buffer
                self.timer_copy_ibuf.start();
                let ibuf_command_buffer = copy_buffer(
                    &self.device,
                    self.queue,
                    self.command_pool,
                    index_staging_buffer,
                    index_buffer,
                    index_buffer_size,
                    ibuf_copied_fence,
                );
                self.timer_copy_ibuf.stop();

                copy_command_buffers = Some((vbuf_command_buffer, ibuf_command_buffer));
            }

            // create command buffer
            self.timer_create_cbuf.start();
            let command_buffer = create_command_buffer(
                &self.device,
                self.render_pass,
                self.pipeline,
                self.command_pool,
                self.framebuffers[image_idx as usize],
                self.swapchain_dims,
                vertex_buffer,
                index_buffer,
                mesh.indices.len() as u32,
            );
            self.timer_create_cbuf.stop();

            self.flying_frames[self.frames_drawn % MAX_FRAMES_IN_FLIGHT].command_buffer =
                Some(command_buffer);

            if let Some((vbuf_cbuf, ibuf_cbuf)) = copy_command_buffers {
                self.timer_copy_complete.start();
                // before we submit, wait for both copy operations to finish
                unsafe {
                    self.device.wait_for_fences(
                        &[vbuf_copied_fence, ibuf_copied_fence],
                        true,
                        std::u64::MAX,
                    )
                }
                .expect("Couldn't wait for vertex and index buffer to finish copying");
                self.timer_copy_complete.stop();

                // free the command buffers used to copy
                unsafe {
                    self.device
                        .free_command_buffers(self.command_pool, &[vbuf_cbuf, ibuf_cbuf])
                };
            }

            // submit command buffer
            let wait_semaphores = [image_available_semaphore];

            // "Each entry in the waitStages array corresponds to the semaphore with
            // the same index in pWaitSemaphores."
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

            let cur_command_buffers = [command_buffer];

            let signal_semaphores = [render_finished_semaphore];

            let submit_info = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                p_next: ptr::null(),
                wait_semaphore_count: 1,
                p_wait_semaphores: wait_semaphores.as_ptr(),
                p_wait_dst_stage_mask: wait_stages.as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: cur_command_buffers.as_ptr(),
                signal_semaphore_count: 1,
                p_signal_semaphores: signal_semaphores.as_ptr(),
            };

            self.timer_draw.start();

            // somebody else was previously using this fence and we waited until it
            // was signalled (operation completed). now we need to reset it, because
            // we aren't yet done but the fence says we are.
            unsafe { self.device.reset_fences(&[render_finished_fence]) }
                .expect("Couldn't reset render_finished_fence");

            let submissions = [submit_info];
            unsafe {
                self.device
                    .queue_submit(self.queue, &submissions, render_finished_fence)
            }
            .expect("Couldn't submit command buffer");

            // present result to swapchain
            let swapchains = [self.swapchain];
            let image_indices = [image_idx];

            let present_info = vk::PresentInfoKHR {
                s_type: vk::StructureType::PRESENT_INFO_KHR,
                p_next: ptr::null(),
                wait_semaphore_count: 1,
                p_wait_semaphores: signal_semaphores.as_ptr(),
                swapchain_count: 1,
                p_swapchains: swapchains.as_ptr(),
                p_image_indices: image_indices.as_ptr(),
                p_results: ptr::null_mut(),
            };

            match unsafe {
                self.swapchain_creator
                    .queue_present(self.queue, &present_info)
            } {
                Ok(_idk_what_this_is) => (),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.must_recreate_swapchain = true;
                    continue;
                }
                Err(e) => panic!("Unexpected error during queue_present: {}", e),
            };

            self.frames_drawn += 1;
        }

        println!("FPS: {:.2}", self.frames_drawn as f64 / get_elapsed(self.start_time));
        println!(
            "Average delta in ms: {:.5}",
            get_elapsed(self.start_time) / self.frames_drawn as f64 * 1_000.0
        );

        self.timer_draw.print();
        self.timer_mesh_wait.print();
        self.timer_write_vbuf.print();
        self.timer_copy_vbuf.print();
        self.timer_write_ibuf.print();
        self.timer_copy_ibuf.print();
        self.timer_copy_complete.print();
        self.timer_create_cbuf.print();
    }

    pub fn cleanup(mut self) {
        unsafe { self.device.device_wait_idle() }.expect("Couldn't wait for device to become idle");

        // destroy objects
        unsafe {
            self.flying_frames.iter().for_each(|f| {
                if let Some(command_buffer) = f.command_buffer {
                    self.device
                        .free_command_buffers(self.command_pool, &[command_buffer]);
                }

                self.device
                    .destroy_semaphore(f.image_available_semaphore, None);
                self.device
                    .destroy_semaphore(f.render_finished_semaphore, None);
                self.device.destroy_fence(f.render_finished_fence, None);

                self.device.destroy_buffer(f.vertex_staging_buffer, None);
                self.device.destroy_buffer(f.vertex_buffer, None);
                self.device.destroy_buffer(f.index_staging_buffer, None);
                self.device.destroy_buffer(f.index_buffer, None);

                self.device
                    .free_memory(f.vertex_staging_buffer_memory, None);
                self.device.free_memory(f.vertex_buffer_memory, None);
                self.device.free_memory(f.index_staging_buffer_memory, None);
                self.device.free_memory(f.index_buffer_memory, None);

                self.device.destroy_fence(f.vbuf_copied_fence, None);
                self.device.destroy_fence(f.ibuf_copied_fence, None);
            });

            cleanup_swapchain(
                &self.device,
                &self.swapchain_creator,
                self.framebuffers,
                self.swapchain_image_views,
                self.swapchain,
            );

            self.device.destroy_pipeline(self.pipeline, None);

            self.device.destroy_command_pool(self.command_pool, None);
        }

        let mesh_recv = self.mesh_recv;
        drop(mesh_recv);
    }
}

fn create_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    dimensions: vk::Extent2D,
    image_views: &[vk::ImageView],
) -> Vec<vk::Framebuffer> {
    image_views
        .iter()
        .map(|iv| {
            let image_views = [*iv];
            let framebuffer_info = vk::FramebufferCreateInfo {
                s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::FramebufferCreateFlags::empty(),
                render_pass,
                attachment_count: 1,
                p_attachments: image_views.as_ptr(),
                width: dimensions.width,
                height: dimensions.height,
                layers: 1,
            };

            unsafe { device.create_framebuffer(&framebuffer_info, None) }
                .expect("Couldn't create framebuffer")
        })
        .collect()
}

fn setup_flying_frames(
    device: &Device,
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    vertex_buffer_capacity: DeviceSize,
    index_buffer_capacity: DeviceSize,
) -> Vec<FlyingFrame> {
    // create info for semaphores and fences
    let semaphore_info = vk::SemaphoreCreateInfo {
        s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::SemaphoreCreateFlags::empty(),
    };

    let fence_info = vk::FenceCreateInfo {
        s_type: vk::StructureType::FENCE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::FenceCreateFlags::SIGNALED,
    };

    (0..MAX_FRAMES_IN_FLIGHT)
        .map(|_| {
            let (vertex_staging_buffer, vertex_staging_buffer_memory) = create_buffer(
                device,
                device_memory_properties,
                vertex_buffer_capacity,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );

            let (index_staging_buffer, index_staging_buffer_memory) = create_buffer(
                device,
                device_memory_properties,
                index_buffer_capacity,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );

            let (vertex_buffer, vertex_buffer_memory) = create_buffer(
                device,
                device_memory_properties,
                vertex_buffer_capacity,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

            let (index_buffer, index_buffer_memory) = create_buffer(
                device,
                device_memory_properties,
                index_buffer_capacity,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

            let image_available_semaphore =
                unsafe { device.create_semaphore(&semaphore_info, None) }
                    .expect("Couldn't create semaphore");

            let render_finished_semaphore =
                unsafe { device.create_semaphore(&semaphore_info, None) }
                    .expect("Couldn't create semaphore");

            let render_finished_fence =
                unsafe { device.create_fence(&fence_info, None) }.expect("Couldn't create fence");

            let vbuf_copied_fence =
                unsafe { device.create_fence(&fence_info, None) }.expect("Couldn't create fence");

            let ibuf_copied_fence =
                unsafe { device.create_fence(&fence_info, None) }.expect("Couldn't create fence");

            FlyingFrame {
                image_available_semaphore,
                render_finished_semaphore,
                render_finished_fence,
                command_buffer: None,

                vertex_staging_buffer,
                vertex_staging_buffer_memory,
                vertex_buffer,
                vertex_buffer_memory,
                index_staging_buffer,
                index_staging_buffer_memory,
                index_buffer,
                index_buffer_memory,

                vbuf_copied_fence,
                ibuf_copied_fence,
            }
        })
        .collect()
}

fn create_buffer(
    device: &Device,
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    bytes: DeviceSize,
    usage: vk::BufferUsageFlags,
    required_memory_properties: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let vertex_buffer_info = vk::BufferCreateInfo {
        s_type: vk::StructureType::BUFFER_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::BufferCreateFlags::empty(),
        size: bytes,
        usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,

        // don't understand why this can be left blank
        queue_family_index_count: 0,
        p_queue_family_indices: ptr::null(),
    };

    let vertex_buffer = unsafe { device.create_buffer(&vertex_buffer_info, None) }
        .expect("Couldn't create vertex buffer");

    let vertex_buffer_memory_requirements =
        unsafe { device.get_buffer_memory_requirements(vertex_buffer) };

    let vertex_buffer_memory_type_index = find_memory_type(
        device_memory_properties,
        vertex_buffer_memory_requirements.memory_type_bits,
        required_memory_properties,
    );

    let vertex_buffer_alloc_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        p_next: ptr::null(),

        // this size can be different from the size in vertex_buffer_info! I
        // think the minimum allocation on the GPU is 256 bytes or something.
        allocation_size: vertex_buffer_memory_requirements.size,
        memory_type_index: vertex_buffer_memory_type_index,
    };

    let vertex_buffer_memory = unsafe { device.allocate_memory(&vertex_buffer_alloc_info, None) }
        .expect("Couldn't allocate vertex buffer device memory");

    unsafe { device.bind_buffer_memory(vertex_buffer, vertex_buffer_memory, 0) }
        .expect("Couldn't bind vertex buffer");

    (vertex_buffer, vertex_buffer_memory)
}

// taken from vulkan-tutorial-rust
fn find_memory_type(
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    required_properties: vk::MemoryPropertyFlags,
) -> u32 {
    for (i, memory_type) in memory_properties.memory_types.iter().enumerate() {
        // this is magic, accept it
        if (type_filter & (1 << i)) > 0 && memory_type.property_flags.contains(required_properties)
        {
            return i as u32;
        }
    }

    panic!("Failed to find suitable memory type!")
}

fn cleanup_swapchain(
    device: &Device,
    swapchain_creator: &Swapchain,
    framebuffers: Vec<vk::Framebuffer>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain: vk::SwapchainKHR,
) {
    unsafe {
        swapchain_image_views
            .iter()
            .for_each(|iv| device.destroy_image_view(*iv, None));
        framebuffers
            .iter()
            .for_each(|fb| device.destroy_framebuffer(*fb, None));
        swapchain_creator.destroy_swapchain(swapchain, None);
    }
}

fn create_command_pool(device: &Device, queue_family_index: u32) -> vk::CommandPool {
    let command_pool_info = vk::CommandPoolCreateInfo {
        s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::CommandPoolCreateFlags::empty(),
        queue_family_index,
    };

    unsafe { device.create_command_pool(&command_pool_info, None) }
        .expect("Couldn't create command pool")
}

fn copy_buffer(
    device: &Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    source: vk::Buffer,
    dest: vk::Buffer,
    bytes: DeviceSize,
    fence: vk::Fence,
) -> vk::CommandBuffer {
    // returns the command buffer used to perform the copy operation, which
    // should be freed after the given fence is signalled

    // fence will be signalled when the copy operation finishes
    let command_buffer_alloc_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: ptr::null(),
        command_pool,
        // Primary: can be submitted to a queue, but not called from other command buffers
        // Secondary: can't be directly submitted, but can be called from other command buffers
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
    };

    let command_buffer = unsafe { device.allocate_command_buffers(&command_buffer_alloc_info) }
        .expect("Couldn't allocate command buffers")[0];

    // begin command buffer
    let command_buffer_begin_info = vk::CommandBufferBeginInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        p_next: ptr::null(),
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        p_inheritance_info: ptr::null(),
    };

    unsafe { device.begin_command_buffer(command_buffer, &command_buffer_begin_info) }
        .expect("Couldn't begin command buffer");

    // copy
    let copy_region = [vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: bytes,
    }];

    unsafe { device.cmd_copy_buffer(command_buffer, source, dest, &copy_region) };

    unsafe { device.end_command_buffer(command_buffer) }.expect("Couldn't record command buffer!");

    let command_buffers = [command_buffer];

    // submit
    let submit_info = [vk::SubmitInfo {
        s_type: vk::StructureType::SUBMIT_INFO,
        p_next: ptr::null(),
        wait_semaphore_count: 0,
        p_wait_semaphores: ptr::null(),
        p_wait_dst_stage_mask: ptr::null(),
        command_buffer_count: 1,
        p_command_buffers: command_buffers.as_ptr(),
        signal_semaphore_count: 0,
        p_signal_semaphores: ptr::null(),
    }];

    unsafe { device.queue_submit(queue, &submit_info, fence) }
        .expect("Couldn't submit command buffer");

    command_buffer
}

fn write_to_cpu_accessible_buffer<D: DeviceV1_0, T>(
    device: &D,
    buffer_memory: vk::DeviceMemory,
    data: &[T],
) {
    let buffer_size = size_of_slice(data);

    unsafe {
        let mapped_memory = device
            .map_memory(buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
            .expect("Couldn't map vertex buffer memory") as *mut T;

        mapped_memory.copy_from_nonoverlapping(data.as_ptr(), data.len());

        device.unmap_memory(buffer_memory);
    }
}

fn create_command_buffer(
    device: &Device,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    framebuffer: vk::Framebuffer,
    dimensions: vk::Extent2D,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    index_count: u32,
) -> vk::CommandBuffer {
    let command_buffer_alloc_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: ptr::null(),
        command_pool,
        // Primary: can be submitted to a queue, but not called from other command buffers
        // Secondary: can't be directly submitted, but can be called from other command buffers
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
    };

    let command_buffer = unsafe { device.allocate_command_buffers(&command_buffer_alloc_info) }
        .expect("Couldn't allocate command buffers")[0];

    // create viewport and scissors
    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: dimensions.width as f32,
        height: dimensions.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];

    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: dimensions,
    }];

    // begin command buffer
    let command_buffer_begin_info = vk::CommandBufferBeginInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        p_next: ptr::null(),
        flags: vk::CommandBufferUsageFlags::empty(),
        p_inheritance_info: ptr::null(),
    };

    unsafe { device.begin_command_buffer(command_buffer, &command_buffer_begin_info) }
        .expect("Couldn't begin command buffer");

    // Set viewports and scissors
    unsafe {
        device.cmd_set_viewport(command_buffer, 0, &viewports);
    }
    unsafe {
        device.cmd_set_scissor(command_buffer, 0, &scissors);
    }

    // Start render pass
    let clear_values = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.5, 0.5, 0.5, 1.0],
        },
    }];

    let render_pass_begin_info = vk::RenderPassBeginInfo {
        s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
        p_next: ptr::null(),
        render_pass,
        framebuffer,
        render_area: vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: dimensions,
        },
        clear_value_count: 1,
        p_clear_values: clear_values.as_ptr(),
    };

    unsafe {
        device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_begin_info,
            vk::SubpassContents::INLINE,
        );

        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);

        // bind vertex buffer
        let vertex_buffers = [vertex_buffer];
        let offsets: [vk::DeviceSize; 1] = [0];
        device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);

        // bind index buffer
        device.cmd_bind_index_buffer(command_buffer, index_buffer, 0, vk::IndexType::UINT32);

        // <index_count> indices, 1 instance, first index 0, vertex offset 0,
        // first instance 0
        device.cmd_draw_indexed(command_buffer, index_count, 1, 0, 0, 0);

        device.cmd_end_render_pass(command_buffer);
    }

    unsafe { device.end_command_buffer(command_buffer) }.expect("Couldn't record command buffer!");

    command_buffer
}

fn create_swapchain(
    physical_device: vk::PhysicalDevice,
    surface_loader: &Surface,
    surface: vk::SurfaceKHR,
    swapchain_creator: &Swapchain,
) -> (vk::SwapchainKHR, Vec<vk::Image>, vk::Extent2D) {
    // check device swapchain capabilties (not just that it has the extension,
    // also formats and stuff like that)
    // also returns what dimensions the swapchain should initially be created at
    let dimensions = check_device_swapchain_caps(surface_loader, physical_device, surface);

    // for now, the format is fixed - might be good to change later.

    // create swapchain
    let create_info = vk::SwapchainCreateInfoKHR {
        s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: vk::SwapchainCreateFlagsKHR::empty(),
        surface: surface,
        min_image_count: 2,
        image_format: SWAPCHAIN_FORMAT,
        image_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        image_extent: dimensions,
        image_array_layers: 1,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: ptr::null(),
        pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode: vk::PresentModeKHR::IMMEDIATE,
        clipped: vk::TRUE,
        old_swapchain: vk::SwapchainKHR::null(),
    };

    let swapchain = unsafe { swapchain_creator.create_swapchain(&create_info, None) }
        .expect("Couldn't create swapchain");

    let images = unsafe { swapchain_creator.get_swapchain_images(swapchain) }
        .expect("Couldn't get swapchain images");

    (swapchain, images, dimensions)
}

fn create_swapchain_image_views(
    device: &Device,
    images: &[vk::Image],
) -> Vec<vk::ImageView> {
    images
        .iter()
        .map(|image| {
            let iv_info = vk::ImageViewCreateInfo {
                s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::ImageViewCreateFlags::empty(),
                image: *image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: SWAPCHAIN_FORMAT,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };

            unsafe { device.create_image_view(&iv_info, None) }
                .expect("Couldn't create image view info")
        })
        .collect()
}

fn check_device_swapchain_caps(
    surface_loader: &Surface,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> vk::Extent2D {
    // returns the current dimensions of the swapchain

    let capabilities = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
    }
    .expect("Couldn't get physical device surface capabilities");

    let formats =
        unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface) }
            .expect("Couldn't get physical device surface formats");

    let present_modes = unsafe {
        surface_loader.get_physical_device_surface_present_modes(physical_device, surface)
    }
    .expect("Couldn't get physical device surface present modes");

    // we will request 3 swapchain images to avoid having to wait while one is
    // being cleared or something, idk exactly
    assert!(capabilities.min_image_count <= 3 && capabilities.max_image_count >= 3);

    formats
        .iter()
        .find(|fmt| fmt.format == vk::Format::B8G8R8A8_UNORM)
        .expect("Swapchain doesn't support B8G8R8A8_UNORM!");

    assert!(present_modes.contains(&vk::PresentModeKHR::IMMEDIATE));

    capabilities.current_extent
}
