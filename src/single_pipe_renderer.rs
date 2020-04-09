use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk::DeviceSize;
use ash::{vk, Device, Instance};

use log::{info, warn};

use winit::{Event, EventsLoop, WindowEvent};

use crate::size_of_slice;

use std::ptr;

pub struct Renderer {
    device: Device,
    flying_frames: Vec<FlyingFrame>,

    events_loop: EventsLoop,
    frames_drawn: usize,
}

#[derive(Clone)]
pub struct Mesh<V> {
    pub vertices: Vec<V>,
    pub indices: Vec<IndexType>,
}

// range from 0.0 .. screen size
// -1.0 .. 1.0
type IndexType = u32;

const MAX_FRAMES_IN_FLIGHT: usize = 4;

const LOG_LEVEL: log::LevelFilter = log::LevelFilter::Trace;

const VBUF_CAPACITY: DeviceSize = 1_000_000;
const IBUF_CAPACITY: DeviceSize = 1_000_000;

struct FlyingFrame {
    id: String,

    device: Device,
    queue: vk::Queue,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,

    mesh_buffer: MeshBuffer,

    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    render_finished_fence: vk::Fence,

    prev_used_command_buffer: Option<vk::CommandBuffer>,
}

// does not use a staging buffer
#[derive(Clone)]
struct MeshBuffer {
    vertex: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    index: vk::Buffer,
    index_memory: vk::DeviceMemory,
}

impl Renderer {
    pub fn new(
        device: Device,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        queue: vk::Queue,
        pipeline: vk::Pipeline,
        render_pass: vk::RenderPass,
        queue_family_index: u32,
        events_loop: EventsLoop,
    ) -> Self {
        setup_logger().expect("Couldn't set up logger");

        println!("Maximum frames in flight: {} ", MAX_FRAMES_IN_FLIGHT);

        // command pool
        let command_pool = create_command_pool(&device, queue_family_index);

        // flying frames (one for each swapchain image)
        let device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let flying_frames = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|idx| {
                FlyingFrame::new(
                    device.clone(),
                    queue,
                    render_pass,
                    pipeline,
                    command_pool,
                    device_memory_properties,
                    format!("{}", idx),
                )
            })
            .collect();

        Self {
            device: device.clone(),
            flying_frames,

            events_loop,
            frames_drawn: 0,
        }
    }

    pub fn get_image_available_semaphore(&self) -> vk::Semaphore {
        self.flying_frames[self.frames_drawn % MAX_FRAMES_IN_FLIGHT].get_image_available_semaphore()
    }

    pub fn draw<V>(&mut self, mesh: &Mesh<V>, framebuffer: vk::Framebuffer, swapchain_dims: vk::Extent2D) -> (Vec<WindowEvent>, vk::Semaphore) {
        // returns a list of all events collected in the frame and a semaphore
        // signalled when rendering is finished

        // assumes the mesh is different every frame, so will always copy its
        // data to the GPU

        info!("EVENT 'New Frame'");

        let events = self.collect_events();

        // wait for chosen FlyingFrame's previous rendering operation to
        // complete
        let ff_idx = self.frames_drawn % MAX_FRAMES_IN_FLIGHT;

        info!("BEGIN 'FF {} Wait'", self.flying_frames[ff_idx].id);
        self.flying_frames[ff_idx].wait();
        info!("END 'FF {} Wait'", self.flying_frames[ff_idx].id);

        // write to buffers
        let buffers = self.flying_frames[ff_idx].get_mesh_buffer();

        info!("BEGIN 'Writing Buffers'");
        write_to_cpu_accessible_buffer(&self.device, buffers.vertex_memory, &mesh.vertices);
        write_to_cpu_accessible_buffer(&self.device, buffers.index_memory, &mesh.indices);

        info!("END 'Writing Buffers'");

        let index_count = mesh.indices.len() as u32;

        info!("BEGIN 'Creation/Submission'");
        let render_finished_semaphore = self.flying_frames[ff_idx].create_and_submit(
            buffers.vertex,
            buffers.index,
            index_count,
            framebuffer,
            swapchain_dims,
        );
        info!("END 'Creation/Submission'");

        self.frames_drawn += 1;

        (events, render_finished_semaphore)
    }

    fn collect_events(&mut self) -> Vec<WindowEvent> {
        // returns a list of unprocessed events

        let mut events = vec![];

        self.events_loop.poll_events(|ev| match ev {
            Event::WindowEvent { event, .. } => events.push(event),
            _ => {}
        });

        events
    }

    pub fn cleanup(self) {
        unsafe { self.device.device_wait_idle() }
            .expect("Couldn't wait for device to become idle before cleanup");

        // all flying frames share the same command pool, so only destroy it
        // once
        let command_pool = self.flying_frames[0].get_command_pool();

        self.flying_frames.into_iter().for_each(|mut ff| {
            ff.wait();
            ff.cleanup();
        });

        unsafe { self.device.destroy_command_pool(command_pool, None) };
    }
}

impl FlyingFrame {
    fn new(
        device: Device,
        queue: vk::Queue,
        render_pass: vk::RenderPass,
        pipeline: vk::Pipeline,
        command_pool: vk::CommandPool,
        device_memory_properties: vk::PhysicalDeviceMemoryProperties,
        id: String,
    ) -> Self {
        // id is used for debug purposes

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

        // create mesh buffer
        let mesh_buffer = MeshBuffer::new(
            &device,
            device_memory_properties,
            VBUF_CAPACITY,
            IBUF_CAPACITY,
        );

        Self {
            id,
            device,
            queue,
            render_pass,
            pipeline,
            command_pool,

            mesh_buffer,

            image_available_semaphore,
            render_finished_semaphore,
            render_finished_fence,

            prev_used_command_buffer: None,
        }
    }

    fn create_and_submit(
        &mut self,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
        index_count: u32,
        framebuffer: vk::Framebuffer,
        dimensions: vk::Extent2D,
    ) -> vk::Semaphore {
        // returns a semaphore signalled when rendering completes

        unsafe { self.device.reset_fences(&[self.render_finished_fence]) }
            .expect("Couldn't reset render_finished_fence");

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

        self.prev_used_command_buffer = Some(command_buffer);

        self.render_finished_semaphore
    }

    fn wait(&mut self) {
        unsafe {
            self.device
                .wait_for_fences(&[self.render_finished_fence], true, std::u64::MAX)
        }
        .expect("Couldn't wait for previous rendering operation to finish");

        if let Some(cbuf) = self.prev_used_command_buffer {
            let command_buffers = [cbuf];
            unsafe {
                self.device
                    .free_command_buffers(self.command_pool, &command_buffers)
            };
        } else {
            warn!("No prev cbuf for flying frame {}", self.id);
        }
    }

    fn get_image_available_semaphore(&self) -> vk::Semaphore {
        // returns the semaphore that should be signalled when an image is
        // acquired from the swapchain
        self.image_available_semaphore
    }

    fn get_mesh_buffer(&self) -> MeshBuffer {
        // There is no safeguard to prevent you from using the MeshBuffer when
        // you shouldn't, so be careful
        self.mesh_buffer.clone()
    }

    fn cleanup(self) {
        self.mesh_buffer.cleanup(&self.device);

        unsafe {
            self.device
                .destroy_semaphore(self.image_available_semaphore, None);
            self.device
                .destroy_semaphore(self.render_finished_semaphore, None);
            self.device.destroy_fence(self.render_finished_fence, None);
        }
    }

    fn get_command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }
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

    fn cleanup(self, device: &Device) {
        unsafe {
            device.free_memory(self.vertex_memory, None);
            device.free_memory(self.index_memory, None);
            device.destroy_buffer(self.vertex, None);
            device.destroy_buffer(self.index, None);
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
        .chain(
            fern::Dispatch::new()
                .level(LOG_LEVEL)
                .chain(fern::log_file("single-pipe-renderer.log")?),
        )
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
                .chain(std::io::stdout()),
        )
        .apply()?;

    Ok(())
}
