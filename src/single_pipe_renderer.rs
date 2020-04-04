use ash::extensions::khr::{Swapchain, XlibSurface};
use ash::extensions::{ext::DebugUtils, khr::Surface};
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk::DeviceSize;
use ash::{vk, Device, Instance};

use crossbeam_channel::{Receiver, Sender};

use log::{debug, error, info, trace, warn};

use winit::{dpi::LogicalPosition, ElementState, Event, EventsLoop, MouseButton, WindowEvent};

use crate::{get_elapsed, size_of_slice, LoopTimer};

use std::ptr;

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
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    flying_frames: Vec<FlyingFrame>,
    mesh_buffer_ring: MeshBufferRing,

    events_loop: EventsLoop,
    mesh_recv: Receiver<Mesh<V>>,
    event_send: Sender<WindowEvent>,

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
const MESH_BUFFER_RING_SIZE: usize = 4;
const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;
const MAX_POINT_DIST: f64 = 10.0;

const LOG_LEVEL: log::LevelFilter = log::LevelFilter::Trace;

const VBUF_CAPACITY: DeviceSize = 1_000_000;
const IBUF_CAPACITY: DeviceSize = 1_000_000;

struct FlyingFrame {
    device: Device,
    queue: vk::Queue,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    swapchain_creator: Swapchain,
    swapchain: vk::SwapchainKHR,
    command_pool: vk::CommandPool,

    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    render_finished_fence: vk::Fence,
}

struct MeshBufferRing {
    mesh_buffers: Vec<MeshBuffer>,
    in_use_fences: Vec<Option<vk::Fence>>,
    counter: usize,
}

// does not use a staging buffer
#[derive(Clone)]
struct MeshBuffer {
    vertex: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    index: vk::Buffer,
    index_memory: vk::DeviceMemory,
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
        setup_logger();

        println!("Maximum frames in flight: {} ", MAX_FRAMES_IN_FLIGHT);

        // framebuffer creation
        let framebuffers =
            create_framebuffers(&device, render_pass, swapchain_dims, &swapchain_image_views);

        // flying frames (one for each swapchain image)
        let device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // command pool
        let command_pool = create_command_pool(&device, queue_family_index);

        let flying_frames = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                FlyingFrame::new(
                    device.clone(),
                    queue,
                    render_pass,
                    pipeline,
                    swapchain_creator.clone(),
                    swapchain,
                    command_pool,
                )
            })
            .collect();

        let mesh_buffer_ring = MeshBufferRing::new(&device, device_memory_properties, VBUF_CAPACITY, IBUF_CAPACITY, MESH_BUFFER_RING_SIZE);
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
            device_memory_properties,
            flying_frames,
            mesh_buffer_ring,

            events_loop,
            mesh_recv,
            event_send,

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
            info!("Begin new frame");

            if self.must_recreate_swapchain {
                self.recreate_swapchain();
            }

            if self.process_events() {
                break;
            }

            // wait for new mesh to be sent
            info!("Waiting for mesh...");
            let mesh = self.mesh_recv.recv().unwrap();
            info!("Got mesh");

            // wait for chosen FlyingFrame's previous rendering operation to
            // complete
            let ff_idx = self.frames_drawn % MAX_FRAMES_IN_FLIGHT;

            info!("Waiting for FF to become available...");
            self.flying_frames[ff_idx].wait();
            info!("Done waiting for FF");

            // write buffers
            let render_finished_fence = self.flying_frames[ff_idx].render_finished_fence;

            info!("Acquiring buffer...");
            let buffers = self.mesh_buffer_ring.get(&self.device, render_finished_fence);
            info!("Done acquiring buffer");

            // write to buffers
            info!("Writing buffers...");
            write_to_cpu_accessible_buffer(&self.device, buffers.vertex_memory, &mesh.vertices);
            write_to_cpu_accessible_buffer(&self.device, buffers.index_memory, &mesh.indices);

            info!("Done writing buffers");

            let index_count = mesh.indices.len() as u32;

            let image_available_semaphore =
                self.flying_frames[ff_idx].get_image_available_semaphore();

            let image_index = self.acquire_image(image_available_semaphore);
            let framebuffer = self.framebuffers[image_index as usize];

            info!("Drawing...");
            self.flying_frames[ff_idx].draw(
                image_index,
                buffers.vertex,
                buffers.index,
                index_count,
                framebuffer,
                self.swapchain_dims,
            );
        }

        println!(
            "FPS: {:.2}",
            self.frames_drawn as f64 / get_elapsed(self.start_time)
        );
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

    fn recreate_swapchain(&mut self) {
        unsafe { self.device.device_wait_idle() }
            .expect("Couldn't wait for self.device to become idle");

        // Cleanup_swapchain requires ownership of these to destroy
        // them. They will be re-created later. Technically I think it
        // would also work to pass pointers to cleanup_swapchain instead
        // of transferring ownership, but I don't like the idea of that
        // because then values in command_buffers and framebuffers would
        // point to destroyed vulkan objects.
        let our_framebuffers = std::mem::replace(&mut self.framebuffers, vec![]);
        let our_swapchain_image_views = std::mem::replace(&mut self.swapchain_image_views, vec![]);

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
        self.swapchain_image_views =
            create_swapchain_image_views(&self.device, &self.swapchain_images);

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

    fn process_events(&mut self) -> bool {
        // returns whether a quit event has been received

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

        exit
    }

    fn acquire_image(&mut self, semaphore: vk::Semaphore) -> u32 {
        // takes a semaphore to signal, returns the acquired image index
        let acquire_result = unsafe {
            self.swapchain_creator.acquire_next_image(
                self.swapchain,
                std::u64::MAX,
                semaphore,
                vk::Fence::null(),
            )
        };

        match acquire_result {
            Ok((image_idx, _is_sub_optimal)) => image_idx,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                panic!("Swapchain out of date!");
            }
            Err(e) => panic!("Unexpected error during acquire_next_image: {}", e),
        }
    }
}

impl FlyingFrame {
    pub fn new(
        device: Device,
        queue: vk::Queue,
        render_pass: vk::RenderPass,
        pipeline: vk::Pipeline,
        swapchain_creator: Swapchain,
        swapchain: vk::SwapchainKHR,
        command_pool: vk::CommandPool,
    ) -> Self {
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

        let image_available_semaphore = unsafe { device.create_semaphore(&semaphore_info, None) }
            .expect("Couldn't create semaphore");

        let render_finished_semaphore = unsafe { device.create_semaphore(&semaphore_info, None) }
            .expect("Couldn't create semaphore");

        let render_finished_fence =
            unsafe { device.create_fence(&fence_info, None) }.expect("Couldn't create fence");

        Self {
            device,
            queue,
            render_pass,
            pipeline,
            swapchain_creator,
            swapchain,
            command_pool,

            image_available_semaphore,
            render_finished_semaphore,
            render_finished_fence,
        }
    }

    pub fn draw(
        &mut self,
        image_index: u32,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
        index_count: u32,
        framebuffer: vk::Framebuffer,
        dimensions: vk::Extent2D,
    ) {
        let command_buffer = create_command_buffer(
            &self.device,
            self.render_pass,
            self.pipeline,
            self.command_pool,
            framebuffer,
            dimensions,
            vertex_buffer,
            index_buffer,
            index_count,
        );

        let wait_semaphores = [self.image_available_semaphore];

        // "Each entry in the waitStages array corresponds to the semaphore with
        // the same index in pWaitSemaphores."
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let signal_semaphores = [self.render_finished_semaphore];

        let command_buffers = [command_buffer];

        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: command_buffers.as_ptr(),
            signal_semaphore_count: 1,
            p_signal_semaphores: signal_semaphores.as_ptr(),
        };

        let submissions = [submit_info];
        unsafe {
            self.device
                .queue_submit(self.queue, &submissions, self.render_finished_fence)
        }
        .expect("Couldn't submit command buffer");

        // present
        let swapchains = [self.swapchain];
        let image_indices = [image_index];

        let signal_semaphores = [self.render_finished_semaphore];

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
                panic!("Swapchain out of date!");
            }
            Err(e) => panic!("Unexpected error during queue_present: {}", e),
        };
    }

    pub fn wait(&mut self) {
        unsafe {
            self.device
                .wait_for_fences(&[self.render_finished_fence], true, std::u64::MAX)
        }
        .expect("Couldn't wait for previous rendering operation to finish");

        unsafe { self.device.reset_fences(&[self.render_finished_fence]) }
            .expect("Couldn't reset render_finished_fence");
    }

    pub fn get_image_available_semaphore(&self) -> vk::Semaphore {
        // returns the semaphore that should be signalled whenn an image is
        // acquired from the swapchain
        self.image_available_semaphore
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

fn create_buffer(
    device: &Device,
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    bytes: DeviceSize,
    usage: vk::BufferUsageFlags,
    required_memory_properties: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let buffer_info = vk::BufferCreateInfo {
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

    let buffer =
        unsafe { device.create_buffer(&buffer_info, None) }.expect("Couldn't create buffer");

    let buffer_memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let buffer_memory_type_index = find_memory_type(
        device_memory_properties,
        buffer_memory_requirements.memory_type_bits,
        required_memory_properties,
    );

    let buffer_alloc_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        p_next: ptr::null(),

        // this size can be different from the size in buffer_info! I
        // think the minimum allocation on the GPU is 256 bytes or something.
        allocation_size: buffer_memory_requirements.size,
        memory_type_index: buffer_memory_type_index,
    };

    let buffer_memory = unsafe { device.allocate_memory(&buffer_alloc_info, None) }
        .expect("Couldn't allocate buffer device memory");

    unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0) }.expect("Couldn't bind buffer");

    (buffer, buffer_memory)
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

fn write_to_cpu_accessible_buffer<T>(device: &Device, buffer_memory: vk::DeviceMemory, data: &[T]) {
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

fn create_swapchain_image_views(device: &Device, images: &[vk::Image]) -> Vec<vk::ImageView> {
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

impl MeshBufferRing {
    fn new(
        device: &Device,
        device_memory_properties: vk::PhysicalDeviceMemoryProperties,
        vbuf_capacity: vk::DeviceSize,
        ibuf_capacity: vk::DeviceSize,
        buffer_count: usize,
    ) -> Self {
        Self {
            mesh_buffers:
            (0..buffer_count)
                .map(|_| MeshBuffer::new(
                    device,
                    device_memory_properties,
                    vbuf_capacity,
                    ibuf_capacity,
                ))
                .collect(),
            counter: 0,
            in_use_fences: vec![None; buffer_count],
        }
    }

    fn get(&mut self, device: &Device, done_fence: vk::Fence) -> MeshBuffer {
        // done_fence should be signalled once the buffers are no longer in use

        let index = self.counter % self.mesh_buffers.len();

        // wait on the fence of the mesh buffer we're trying to acquire, if it
        // exists
        match self.in_use_fences[index] {
            Some(fence) =>
                unsafe {
                    device
                        .wait_for_fences(&[fence], true, std::u64::MAX)
                }
                .expect("Couldn't wait for mesh buffer fence to be signalled"),
            None => warn!("No done_fence on meshbuf acquire #{}!", self.counter),
        };

        self.in_use_fences[index] = None;

        self.counter += 1;

        self.mesh_buffers[index].clone()
    }
}

impl MeshBuffer {
    fn new(
        device: &Device,
        device_memory_properties: vk::PhysicalDeviceMemoryProperties,
        vbuf_capacity: vk::DeviceSize,
        ibuf_capacity: vk::DeviceSize,
    ) -> Self {
        let (vertex, vertex_memory) = create_buffer(
            device,
            device_memory_properties,
            vbuf_capacity,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let (index, index_memory) = create_buffer(
            device,
            device_memory_properties,
            ibuf_capacity,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        Self {
            vertex,
            vertex_memory,
            index,
            index_memory,
        }
    }
}


fn setup_logger() -> Result<(), fern::InitError> {
    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{}][{}] {}",
                chrono::Local::now().timestamp_nanos(),
                record.level(),
                message,
            ))
        })
        .level(LOG_LEVEL)
        .chain(fern::log_file("single-pipe-renderer.log")?)
        .chain(
    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        .level(log::LevelFilter::Warn)
        .chain(std::io::stdout())
    )
    .apply()?;

    Ok(())
}
