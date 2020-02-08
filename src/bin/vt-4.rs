use ash::extensions::khr::{Swapchain, XlibSurface};
use ash::extensions::{ext::DebugUtils, khr::Surface};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::{vk, vk_make_version, Entry};

use std::convert::TryInto;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr;

use winit::{Event, WindowEvent};

const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;
const FRAMES_IN_FLIGHT: usize = 5;

pub fn main() {
    // create winit window
    let mut events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new()
        .with_title("Ash - Example")
        .build(&events_loop)
        .unwrap();

    // create instance
    let app_info = vk::ApplicationInfo {
        s_type: vk::StructureType::APPLICATION_INFO,
        p_next: ptr::null(),
        p_application_name: CString::new("Thingy").unwrap().as_ptr(),
        application_version: 0,
        p_engine_name: CString::new("Vulkan Engine").unwrap().as_ptr(),
        engine_version: 0,
        api_version: vk_make_version!(1, 1, 0),
    };

    let layer_names = [
        CString::new("VK_LAYER_LUNARG_standard_validation").unwrap(),
        CString::new("VK_LAYER_KHRONOS_validation").unwrap(),
    ];

    let layers_names_raw: Vec<*const i8> = layer_names
        .iter()
        .map(|raw_name| raw_name.as_ptr())
        .collect();

    let extension_names_raw = extension_names();

    let debug_utils_create_info = vk::DebugUtilsMessengerCreateInfoEXT {
        s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        p_next: ptr::null(),
        flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
            | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
            // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(vulkan_debug_utils_callback),
        p_user_data: ptr::null_mut(),
    };

    let create_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        // setting this to a null pointer also works, but leave it like this to
        // be safe i guess?
        p_next: &debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT
            as *const c_void,
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: &app_info,
        enabled_layer_count: layer_names.len() as u32,
        pp_enabled_layer_names: layers_names_raw.as_ptr(),
        enabled_extension_count: extension_names_raw.len() as u32,
        pp_enabled_extension_names: extension_names_raw.as_ptr(),
    };

    let entry = Entry::new().unwrap();

    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("Couldn't create instance")
    };

    let debug_utils_loader = ash::extensions::ext::DebugUtils::new(&entry, &instance);

    let debug_utils_messenger = unsafe {
        debug_utils_loader
            .create_debug_utils_messenger(&debug_utils_create_info, None)
            .expect("Debug Utils Callback")
    };

    // create surface
    let surface =
        unsafe { create_surface(&entry, &instance, &window) }.expect("couldn't create surface");

    let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);

    // get physical device
    let physical_device = {
        let phys_devs = unsafe { instance.enumerate_physical_devices() }
            .expect("Couldn't enumerate physical devices");

        *phys_devs
            .iter()
            .find(|phys_dev| is_phys_dev_suitable(&instance, phys_dev))
            .expect("No suitable physical device found!")
    };

    // get queue family index
    let queue_family_index: u32 = {
        let queues =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        queues
            .iter()
            .enumerate()
            .find(|(_idx, queue)| queue.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .expect("Couldn't find a graphics queue")
            .0
            .try_into()
            .unwrap()
    };

    if !unsafe {
        surface_loader.get_physical_device_surface_support(
            physical_device,
            queue_family_index,
            surface,
        )
    } {
        panic!("Queue does not have surface support! It's possible that a separate queue with surface support exists, but the current implementation is not capable of finding one.");
    }

    // get logical device
    let device_queue_create_info = vk::DeviceQueueCreateInfo {
        s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::DeviceQueueCreateFlags::empty(),
        queue_family_index,
        queue_count: 1,
        p_queue_priorities: [1.0].as_ptr(),
    };

    let device_extensions_raw = get_device_extensions_raw();

    let device_create_info = vk::DeviceCreateInfo {
        s_type: vk::StructureType::DEVICE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::DeviceCreateFlags::empty(),
        queue_create_info_count: 1,
        p_queue_create_infos: [device_queue_create_info].as_ptr(),
        // not used by Vulkan anymore
        enabled_layer_count: 0,
        pp_enabled_layer_names: ptr::null(),
        // these are
        enabled_extension_count: device_extensions_raw.len() as u32,
        pp_enabled_extension_names: device_extensions_raw.as_ptr(),
        p_enabled_features: &vk::PhysicalDeviceFeatures::builder().build(),
    };

    let device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .expect("Couldn't create device")
    };

    // get queue (0 = take first queue)
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    // check device swapchain capabilties (not just that it has the extension,
    // also formats and stuff like that)
    // also returns what dimensions the swapchain should initially be created at
    let starting_dims = check_device_swapchain_caps(&surface_loader, physical_device, surface);
    dbg![starting_dims];

    // create swapchain
    let swapchain_create_info = vk::SwapchainCreateInfoKHR {
        s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: vk::SwapchainCreateFlagsKHR::empty(),
        surface: surface,
        min_image_count: 4,
        image_format: SWAPCHAIN_FORMAT,
        image_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        image_extent: starting_dims,
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

    let swapchain_creator = Swapchain::new(&instance, &device);
    let swapchain = unsafe { swapchain_creator.create_swapchain(&swapchain_create_info, None) }
        .expect("Couldn't create swapchain");

    let images = unsafe { swapchain_creator.get_swapchain_images(swapchain) }
        .expect("Couldn't get swapchain images");

    println!("Swapchain image count: {}", images.len());

    let image_views: Vec<_> = images
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
        .collect();

    dbg![image_views.len()];

    // shaders
    let frag_code = read_shader_code(&relative_path("shaders/vt-3/triangle.frag.spv"));
    let vert_code = read_shader_code(&relative_path("shaders/vt-3/triangle.vert.spv"));

    let frag_module = create_shader_module(&device, frag_code);
    let vert_module = create_shader_module(&device, vert_code);

    let entry_point = CString::new("main").unwrap();

    let vert_stage_info = vk::PipelineShaderStageCreateInfo {
        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineShaderStageCreateFlags::empty(),
        stage: vk::ShaderStageFlags::VERTEX,
        module: vert_module,
        p_name: entry_point.as_ptr(),
        p_specialization_info: &vk::SpecializationInfo::default(),
    };

    let frag_stage_info = vk::PipelineShaderStageCreateInfo {
        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineShaderStageCreateFlags::empty(),
        stage: vk::ShaderStageFlags::FRAGMENT,
        module: frag_module,
        p_name: entry_point.as_ptr(),
        p_specialization_info: &vk::SpecializationInfo::default(),
    };

    let shader_stages = [vert_stage_info, frag_stage_info];

    // fixed-function pipeline settings

    // a.k.a vertex format
    // we don't really have a format since they are hard-coded into the vertex
    // shader for now
    let pipeline_vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

    let pipeline_input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        primitive_restart_enable: vk::FALSE,
    };

    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: starting_dims.width as f32,
        height: starting_dims.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];

    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: starting_dims,
    }];

    let viewport_state = vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineViewportStateCreateFlags::empty(),
        viewport_count: viewports.len() as u32,
        p_viewports: viewports.as_ptr(),
        scissor_count: scissors.len() as u32,
        p_scissors: scissors.as_ptr(),
    };

    let pipeline_rasterization_info = vk::PipelineRasterizationStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineRasterizationStateCreateFlags::empty(),
        depth_clamp_enable: vk::FALSE,
        rasterizer_discard_enable: vk::FALSE,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::NONE,
        front_face: vk::FrontFace::CLOCKWISE,
        depth_bias_enable: vk::FALSE,
        depth_bias_constant_factor: 0.0,
        depth_bias_clamp: 0.0,
        depth_bias_slope_factor: 0.0,
        line_width: 1.0,
    };

    let pipeline_multisample_info = vk::PipelineMultisampleStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineMultisampleStateCreateFlags::empty(),
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        sample_shading_enable: vk::FALSE,
        min_sample_shading: 1.0,
        p_sample_mask: ptr::null(),
        alpha_to_coverage_enable: vk::FALSE,
        alpha_to_one_enable: vk::FALSE,
    };

    // color blending info per framebuffer
    let pipeline_color_blend_attachment_infos = [vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE,
        // not used because we disabled blending
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,

        // is used
        color_write_mask: vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B,
    }];

    // color blending settings for the whole pipleine
    let pipeline_color_blend_info = vk::PipelineColorBlendStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineColorBlendStateCreateFlags::empty(),
        logic_op_enable: vk::FALSE,
        logic_op: vk::LogicOp::COPY, // optional
        attachment_count: pipeline_color_blend_attachment_infos.len() as u32,
        p_attachments: pipeline_color_blend_attachment_infos.as_ptr(),
        blend_constants: [0.0, 0.0, 0.0, 0.0], // optional
    };

    // we don't use any shader uniforms so we leave it empty
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineLayoutCreateFlags::empty(),
        set_layout_count: 0,
        p_set_layouts: ptr::null(),
        push_constant_range_count: 0,
        p_push_constant_ranges: ptr::null(),
    };

    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .expect("Couldn't create pipeline layout!")
    };

    // render pass
    let render_pass = create_render_pass(&device);

    // pipeline
    let pipeline_infos = [vk::GraphicsPipelineCreateInfo {
        s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineCreateFlags::empty(),
        stage_count: shader_stages.len() as u32,
        p_stages: shader_stages.as_ptr(),
        p_vertex_input_state: &pipeline_vertex_input_info,
        p_input_assembly_state: &pipeline_input_assembly_info,
        p_tessellation_state: ptr::null(),
        p_viewport_state: &viewport_state,
        p_rasterization_state: &pipeline_rasterization_info,
        p_multisample_state: &pipeline_multisample_info,
        p_depth_stencil_state: ptr::null(),
        p_color_blend_state: &pipeline_color_blend_info,
        p_dynamic_state: ptr::null(),
        layout: pipeline_layout,
        render_pass,
        subpass: 0,
        base_pipeline_handle: vk::Pipeline::null(),
        base_pipeline_index: 0,
    }];

    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
    }
    .expect("Couldn't create graphics pipeline")[0];

    // shader modules only need to live long enough to create the pipeline
    unsafe {
        device.destroy_shader_module(frag_module, None);
        device.destroy_shader_module(vert_module, None);
    }

    // framebuffer creation
    let framebuffers = create_framebuffers(&device, render_pass, starting_dims, &image_views);

    // command pool
    let command_pool = create_command_pool(&device, queue_family_index);

    // command buffers (re-used between frames)
    let command_buffers = create_command_buffers(
        &device,
        render_pass,
        command_pool,
        starting_dims,
        &framebuffers,
        pipeline,
    );

    // sync objects
    let (
        image_available_semaphores,
        render_finished_semaphores,
        in_flight_fences,
        mut images_in_flight,
    ) = create_sync_objects(&device, image_views.len());

    // will also be used to keep track of which set of synchronization
    // primitives to use
    let mut frames_drawn = 0;

    // used to calculate FPS
    let start_time = std::time::Instant::now();

    loop {
        let mut exit = false;

        events_loop.poll_events(|ev| match ev {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => exit = true,
            _ => {}
        });

        if exit {
            break;
        }

        let cur_image_available_semaphore =
            image_available_semaphores[frames_drawn % FRAMES_IN_FLIGHT];
        let cur_render_finished_semaphore =
            render_finished_semaphores[frames_drawn % FRAMES_IN_FLIGHT];
        let cur_fence = in_flight_fences[frames_drawn % FRAMES_IN_FLIGHT];

        // wait for the fence associated with this command buffer
        unsafe { device.wait_for_fences(&[cur_fence], true, std::u64::MAX) }
            .expect("Couldn't wait for in_flight fence");

        let (image_idx, _is_sub_optimal) = unsafe {
            swapchain_creator.acquire_next_image(
                swapchain,
                std::u64::MAX,
                cur_image_available_semaphore,
                vk::Fence::null(),
            )
        }
        .expect("Couldn't acquire next image");

        // make sure the image we just acquired is not in flight
        if let Some(image_fence) = images_in_flight[image_idx as usize] {
            unsafe { device.wait_for_fences(&[image_fence], true, std::u64::MAX) }
                .expect("Couldn't wait for image_in_flight fence");
        }

        images_in_flight[image_idx as usize] = Some(cur_fence);

        // submit command buffer
        let wait_semaphores = [cur_image_available_semaphore];

        // "Each entry in the waitStages array corresponds to the semaphore with
        // the same index in pWaitSemaphores."
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let cur_command_buffers = [command_buffers[image_idx as usize]];

        let signal_semaphores = [cur_render_finished_semaphore];

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

        // reset fence
        unsafe { device.reset_fences(&[cur_fence]) }.expect("Couldn't reset in_flight fence");

        let submissions = [submit_info];
        unsafe { device.queue_submit(queue, &submissions, cur_fence) }
            .expect("Couldn't submit command buffer");

        // present result to swapchain
        let swapchains = [swapchain];
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

        unsafe { swapchain_creator.queue_present(queue, &present_info) }
            .expect("Couldn't present swapchain result");

        frames_drawn += 1;
    }

    println!("FPS: {:.2}", frames_drawn as f64 / get_elapsed(start_time));
    println!("Average delta in ms: {:.5}", get_elapsed(start_time) / frames_drawn as f64 * 1_000.0);

    unsafe { device.device_wait_idle() }.expect("Couldn't wait for device to become idle");

    // destroy objects
    unsafe {
        image_views
            .iter()
            .for_each(|iv| device.destroy_image_view(*iv, None));
        framebuffers
            .iter()
            .for_each(|fb| device.destroy_framebuffer(*fb, None));
        image_available_semaphores
            .iter()
            .for_each(|sem| device.destroy_semaphore(*sem, None));
        render_finished_semaphores
            .iter()
            .for_each(|sem| device.destroy_semaphore(*sem, None));
        in_flight_fences
            .iter()
            .for_each(|fence| device.destroy_fence(*fence, None));
        device.destroy_command_pool(command_pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        swapchain_creator.destroy_swapchain(swapchain, None);
        device.destroy_render_pass(render_pass, None);
        device.destroy_device(None);
        surface_loader.destroy_surface(surface, None);
        debug_utils_loader.destroy_debug_utils_messenger(debug_utils_messenger, None);
        instance.destroy_instance(None);
    }
}

fn create_sync_objects<D: DeviceV1_0>(
    device: &D,
    swapchain_image_count: usize,
) -> (
    Vec<vk::Semaphore>,
    Vec<vk::Semaphore>,
    Vec<vk::Fence>,
    Vec<Option<vk::Fence>>,
) {
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

    let image_available_semaphores = (0..FRAMES_IN_FLIGHT)
        .map(|_| {
            unsafe { device.create_semaphore(&semaphore_info, None) }
                .expect("Couldn't create semaphore")
        })
        .collect();

    let render_finished_semaphores = (0..FRAMES_IN_FLIGHT)
        .map(|_| {
            unsafe { device.create_semaphore(&semaphore_info, None) }
                .expect("Couldn't create semaphore")
        })
        .collect();

    let in_flight_fences = (0..FRAMES_IN_FLIGHT)
        .map(|_| unsafe { device.create_fence(&fence_info, None) }.expect("Couldn't create fence"))
        .collect();

    let images_in_flight = (0..swapchain_image_count).map(|_| None).collect();

    (
        image_available_semaphores,
        render_finished_semaphores,
        in_flight_fences,
        images_in_flight,
    )
}

fn create_command_pool<D: DeviceV1_0>(device: &D, queue_family_index: u32) -> vk::CommandPool {
    let command_pool_info = vk::CommandPoolCreateInfo {
        s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::CommandPoolCreateFlags::empty(),
        queue_family_index,
    };

    unsafe { device.create_command_pool(&command_pool_info, None) }
        .expect("Couldn't create command pool")
}

fn create_command_buffers<D: DeviceV1_0>(
    device: &D,
    render_pass: vk::RenderPass,
    command_pool: vk::CommandPool,
    dimensions: vk::Extent2D,
    framebuffers: &[vk::Framebuffer],
    pipeline: vk::Pipeline,
) -> Vec<vk::CommandBuffer> {
    let command_buffer_alloc_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: ptr::null(),
        command_pool,
        // Primary: can be submitted to a queue, but not called from other command buffers
        // Secondary: can't be directly submitted, but can be called from other command buffers
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: framebuffers.len() as u32,
    };

    let command_buffers = unsafe { device.allocate_command_buffers(&command_buffer_alloc_info) }
        .expect("Couldn't allocate command buffers");

    command_buffers
        .iter()
        .enumerate()
        .map(|(idx, &command_buffer)| {
            // begin command buffer
            let command_buffer_begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: vk::CommandBufferUsageFlags::empty(),
                p_inheritance_info: ptr::null(),
            };

            unsafe { device.begin_command_buffer(command_buffer, &command_buffer_begin_info) }
                .expect("Couldn't begin command buffer");

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
                framebuffer: framebuffers[idx],
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

                // 3 vertices, 1 instance, first vertex 0, first instance 0
                device.cmd_draw(command_buffer, 3, 1, 0, 0);

                device.cmd_end_render_pass(command_buffer);
            }

            unsafe { device.end_command_buffer(command_buffer) }
                .expect("Couldn't record command buffer!");

            command_buffer
        })
        .collect()
}

fn create_framebuffers<D: DeviceV1_0>(
    device: &D,
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

fn create_shader_module<D: DeviceV1_0>(device: &D, code: Vec<u8>) -> vk::ShaderModule {
    use ash::util::read_spv;
    use std::io::Cursor;

    let readable_code = read_spv(&mut Cursor::new(&code)).expect("Couldn't read SPV");
    let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(&readable_code);

    unsafe {
        device
            .create_shader_module(&shader_module_create_info, None)
            .expect("Couldn't create shader module")
    }
}

fn extension_names() -> Vec<*const i8> {
    // these are instance extensions
    vec![
        Surface::name().as_ptr(),
        XlibSurface::name().as_ptr(),
        DebugUtils::name().as_ptr(),
    ]
}

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);
    eprintln!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}

fn is_phys_dev_suitable(instance: &ash::Instance, phys_dev: &vk::PhysicalDevice) -> bool {
    // gets a list of extensions supported by this device as vulkan strings,
    // which don't implement PartialEq
    let extension_properties = unsafe { instance.enumerate_device_extension_properties(*phys_dev) }
        .expect("Couldn't enumerate device extension properties!");

    // Now convert them into rust strings
    let available_extension_names: Vec<String> = extension_properties
        .iter()
        .map(|ext| vk_to_string(&ext.extension_name))
        .collect();

    // make sure all required device extensions are supported by this device
    get_device_extensions().iter().for_each(|name| {
        available_extension_names
            .iter()
            .find(|ext| ext == name)
            .expect(&format!("Couldn't find extension {}", name));
    });

    true
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

// many of these functions are ripped from https://github.com/bwasty/vulkan-tutorial-rs

// only works on linux
unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use winit::os::unix::WindowExt;

    let x11_display = window.get_xlib_display().unwrap();
    let x11_window = window.get_xlib_window().unwrap();
    let x11_create_info = vk::XlibSurfaceCreateInfoKHR {
        s_type: vk::StructureType::XLIB_SURFACE_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        window: x11_window as vk::Window,
        dpy: x11_display as *mut vk::Display,
    };
    let xlib_surface_loader = XlibSurface::new(entry, instance);
    xlib_surface_loader.create_xlib_surface(&x11_create_info, None)
}

fn create_render_pass(device: &ash::Device) -> vk::RenderPass {
    // our render pass has a single image, so only one attachment description is
    // necessary
    let attachment_descs = [vk::AttachmentDescription {
        flags: vk::AttachmentDescriptionFlags::empty(),
        format: SWAPCHAIN_FORMAT,
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::CLEAR,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
    }];

    let color_attachments = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];

    let subpass_descs = [vk::SubpassDescription {
        flags: vk::SubpassDescriptionFlags::empty(),
        pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
        input_attachment_count: 0,
        p_input_attachments: ptr::null(),
        color_attachment_count: 1,
        p_color_attachments: color_attachments.as_ptr(),
        p_resolve_attachments: ptr::null(),
        p_depth_stencil_attachment: ptr::null(),
        preserve_attachment_count: 0,
        p_preserve_attachments: ptr::null(),
    }];

    // apparently needed to ensure we don't start drawing before we acquire a
    // swapchain image, but I don't know why it's necessary because we already
    // use a semaphore to synchronize between acquiring the image and executing
    // the command buffer.
    let subpass_dependencies = [vk::SubpassDependency {
        // "The special value VK_SUBPASS_EXTERNAL refers to the implicit subpass
        // before or after the render pass depending on whether it is specified
        // in srcSubpass or dstSubpass"
        src_subpass: vk::SUBPASS_EXTERNAL,

        // refers to the subpass we draw in, which is at index 0
        dst_subpass: 0,

        // i don't understand this
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        src_access_mask: vk::AccessFlags::empty(),
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dependency_flags: vk::DependencyFlags::empty(),
    }];

    let render_pass_info = vk::RenderPassCreateInfo {
        s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::RenderPassCreateFlags::empty(),
        attachment_count: 1,
        p_attachments: attachment_descs.as_ptr(),
        subpass_count: 1,
        p_subpasses: subpass_descs.as_ptr(),
        dependency_count: 1,
        p_dependencies: subpass_dependencies.as_ptr(),
    };

    unsafe {
        device
            .create_render_pass(&render_pass_info, None)
            .expect("Failed to create render pass!")
    }
}

// Helper function to convert [c_char; SIZE] to string
fn vk_to_string(raw_string_array: &[c_char]) -> String {
    // Implementation 2
    let raw_string = unsafe {
        let pointer = raw_string_array.as_ptr();
        CStr::from_ptr(pointer)
    };

    raw_string
        .to_str()
        .expect("Failed to convert vulkan raw string.")
        .to_owned()
}

fn get_device_extensions<'a>() -> [&'a str; 1] {
    ["VK_KHR_swapchain"]
}

fn get_device_extensions_raw() -> [*const c_char; 1] {
    [ash::extensions::khr::Swapchain::name().as_ptr()]
}

fn read_shader_code(shader_path: &Path) -> Vec<u8> {
    use std::fs::File;
    use std::io::Read;

    let spv_file =
        File::open(shader_path).expect(&format!("Failed to find spv file at {:?}", shader_path));
    let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

    bytes_code
}

pub fn relative_path(local_path: &str) -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), local_path].iter().collect()
}

pub fn get_elapsed(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs() as f64 + start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0
}
