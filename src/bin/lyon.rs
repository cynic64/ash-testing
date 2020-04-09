use ash::extensions::khr::XlibSurface;
use ash::extensions::{ext::DebugUtils, khr::Surface};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::{vk, vk_make_version, Entry};

use winit::dpi::LogicalPosition;
use winit::{ElementState, MouseButton, WindowEvent};

use crossbeam_channel::{Receiver, Sender, TrySendError};

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use std::convert::TryInto;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::ptr;

use memoffset::offset_of;

use ash_testing::single_pipe_renderer::{Mesh, Renderer};
use ash_testing::window::Vindow;
use ash_testing::{relative_path, LoopTimer};

#[cfg(target_os = "macos")]
extern crate cocoa;
#[cfg(target_os = "macos")]
extern crate metal;
#[cfg(target_os = "macos")]
extern crate objc;
extern crate winit;
#[cfg(target_os = "macos")]
use ash::extensions::mvk::MacOSSurface;
#[cfg(target_os = "macos")]
use cocoa::appkit::{NSView, NSWindow};
#[cfg(target_os = "macos")]
use cocoa::base::id as cocoa_id;
#[cfg(target_os = "macos")]
use metal::CoreAnimationLayer;
#[cfg(target_os = "macos")]
use objc::runtime::YES;
#[cfg(target_os = "macos")]
use std::mem;

const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

type IndexType = u32;

// 0-capacity, >0-capacity and unbounded channels all have around the same
// throughput according to my very rough benchmarks
const MESH_CHANNEL_CAPACITY: usize = 8;

const KINKINESS: f64 = 0.2;
// number of pixels allowed to be skipped in between points used
const MAX_POINT_DIST: f64 = 10.0;

const DEBUG_GENERATE_POINTS: bool = true;

// range from 0.0 .. screen size
type PixelPos = [f64; 2];

// -1.0 .. 1.0
type VkPos = [f64; 2];

#[repr(C)]
#[derive(Clone)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

pub fn main() {
    // create winit window
    let events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new()
        .with_title("Ash - Example")
        .build(&events_loop)
        .unwrap();
    let hidpi_factor = window.get_hidpi_factor();

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

    // shaders
    let frag_code = read_shader_code(&relative_path("shaders/vt-5-vbuf/triangle.frag.spv"));
    let vert_code = read_shader_code(&relative_path("shaders/vt-5-vbuf/triangle.vert.spv"));

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

    // render pass
    let render_pass = create_render_pass(&device);

    let mut vindow = Vindow::new(
        physical_device,
        device.clone(),
        instance.clone(),
        queue,
        render_pass,
        surface_loader.clone(),
        surface,
    );

    // pipeline layout
    let pipeline_layout = create_pipeline_layout(&device);

    // pipeline
    let binding_descriptions = Vertex::get_binding_descriptions();
    let attribute_descriptions = Vertex::get_attribute_descriptions();

    let pipeline = create_pipeline(
        &device,
        render_pass,
        pipeline_layout,
        shader_stages,
        &binding_descriptions,
        &attribute_descriptions,
    );

    // the spawned thread will send us new meshes
    let (mesh_send, mesh_recv): (Sender<Mesh<Vertex>>, Receiver<Mesh<Vertex>>) =
        crossbeam_channel::bounded(MESH_CHANNEL_CAPACITY);
    // and the rendering thread will send it events. It can then use them to
    // respond to user mouse input
    let (event_send, event_recv): (Sender<WindowEvent>, Receiver<WindowEvent>) =
        crossbeam_channel::unbounded();

    // spawn it
    let mesh_gen_handle = std::thread::spawn(move || {
        // dummy extent value for now
        mesh_thread(mesh_send, event_recv, hidpi_factor, vk::Extent2D {
            width: 1000,
            height: 1000,
        })
    });

    let mut renderer = Renderer::new(
        device.clone(),
        &instance,
        physical_device,
        queue,
        pipeline,
        render_pass,
        queue_family_index,
        events_loop,
    );

    loop {
        // acquire an image
        let image_available_semaphore = renderer.get_image_available_semaphore();
        let (framebuffer, dims) = vindow.acquire(image_available_semaphore);

        // create and submit command buffer
        let mesh = mesh_recv.recv().unwrap();
        let (events, render_finished_semaphore) = renderer.draw(&mesh, framebuffer, dims);

        // present result
        vindow.present(render_finished_semaphore);

        // process events
        let mut must_quit = false;

        events.iter().for_each(|ev| match ev {
            WindowEvent::CloseRequested => must_quit = true,
            _ => event_send.send(ev.clone()).unwrap(),
        });

        if must_quit {
            break;
        }
    }

    renderer.cleanup();

    drop(mesh_recv);
    println!("waiting on mesh thread");
    mesh_gen_handle.join().unwrap();

    unsafe {
        device.destroy_pipeline(pipeline, None);

        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_render_pass(render_pass, None);

        device.destroy_shader_module(frag_module, None);
        device.destroy_shader_module(vert_module, None);

        device.destroy_device(None);

        surface_loader.destroy_surface(surface, None);

        debug_utils_loader.destroy_debug_utils_messenger(debug_utils_messenger, None);

        instance.destroy_instance(None);
    }
}

fn mesh_thread(
    mesh_send: Sender<Mesh<Vertex>>,
    event_recv: Receiver<WindowEvent>,
    hidpi_factor: f64,
    swapchain_dims: vk::Extent2D,
) {
    let mut timer = LoopTimer::new("Mesh generation".to_string());

    let mut points = if DEBUG_GENERATE_POINTS {
        create_debug_points()
    } else {
        vec![]
    };

    let mut mouse_down = false;
    let mut last_used_mouse_pos = [-9999.0, -9999.0];

    loop {
        timer.start();

        event_recv.try_iter().for_each(|ev| match ev {
            WindowEvent::CursorMoved {
                position: LogicalPosition { x, y },
                ..
            } => {
                if mouse_down {
                    let mouse_pos = [x * hidpi_factor, y * hidpi_factor];

                    // only use mouse pos if it's over a certain distance from the
                    // previous position used
                    if pixel_dist(&mouse_pos, &last_used_mouse_pos) > MAX_POINT_DIST {
                        points.push(pixel_to_vk(swapchain_dims, &mouse_pos));
                        last_used_mouse_pos = mouse_pos;
                    }
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => match state {
                ElementState::Pressed => mouse_down = true,
                ElementState::Released => mouse_down = false,
            },
            _ => {}
        });

        let mesh = create_mesh(&points);

        /*
        println!(
            "Total bytes in mesh: {}, vertices: {}, indices: {}",
            size_of_slice(&mesh.vertices) + size_of_slice(&mesh.indices),
            mesh.vertices.len(),
            mesh.indices.len()
        );
        */

        timer.stop();

        match mesh_send.try_send(mesh.clone()) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => {
                println!("Mesh receiver disconnected, mesh gen thread quitting");
                timer.print();
                return;
            }
        }
    }
}

fn create_mesh(points: &[VkPos]) -> Mesh<Vertex> {
    use lyon::math::Point;
    use lyon::path::Path;
    use lyon::tessellation::*;
    use lyon_tessellation::{LineCap, StrokeOptions, StrokeTessellator};

    let mut builder = Path::builder();

    // need at least 2 points
    if points.len() < 2 {
        return Mesh {
            vertices: vec![],
            indices: vec![],
        };
    }

    // linearly extrapolate one point on either end of the point list to so we
    // can actually draw the first and last point, rather than using them as
    // guides for control points
    // extrapolate backwards: a - (b - a) = 2a - b
    let first_point = vec![[
        2.0 * points[0][0] - points[1][0],
        2.0 * points[0][1] - points[1][1],
    ]];
    // extrapolate forwards: b + (b - a) = 2b - a
    let len = points.len();
    let last_point = vec![[
        2.0 * points[len - 1][0] - points[len - 2][0],
        2.0 * points[len - 1][1] - points[len - 2][1],
    ]];

    let points: Vec<_> = first_point
        .iter()
        .chain(points)
        .chain(&last_point)
        .collect();

    builder.move_to(vk_to_point(&points[1]));

    // last point is used as a control point, and penultimate point is joined
    // to, so only go to len - 2
    for i in 1..points.len() - 2 {
        // we join x and y, using w and z to dictate the control points (a and b)
        let x = points[i];
        let y = points[i + 1];
        let w = points[i - 1];
        let z = points[i + 2];

        let a = [
            x[0] + KINKINESS * y[0] - KINKINESS * w[0],
            x[1] + KINKINESS * y[1] - KINKINESS * w[1],
        ];
        let b = [
            y[0] - KINKINESS * z[0] + KINKINESS * x[0],
            y[1] - KINKINESS * z[1] + KINKINESS * x[1],
        ];

        builder.cubic_bezier_to(vk_to_point(&a), vk_to_point(&b), vk_to_point(&y));
    }

    let path = builder.build();

    // will contain the result of the tessellation.
    let mut geometry: VertexBuffers<Vertex, IndexType> = VertexBuffers::new();
    let mut tessellator = StrokeTessellator::new();
    {
        // compute the tessellation.
        tessellator
            .tessellate(
                &path,
                &StrokeOptions::DEFAULT
                    .with_line_width(0.01)
                    .with_tolerance(0.0001)
                    .with_start_cap(LineCap::Round)
                    .with_end_cap(LineCap::Round),
                &mut BuffersBuilder::new(&mut geometry, |pos: Point, _: StrokeAttributes| Vertex {
                    position: pos.to_array(),
                    color: [
                        pos.x * 0.5 + 0.5,
                        pos.y * 0.5 + 0.5,
                        (pos.x * 0.5 + 0.5) * (pos.y * 0.5 + 0.5),
                    ],
                }),
            )
            .unwrap();
    }

    Mesh {
        vertices: geometry.vertices,
        indices: geometry.indices,
    }
}

fn create_debug_points() -> Vec<VkPos> {
    // creates a list of randomly-generated points from the same seed every
    // time, so that performance of the program can be judged more easily

    let mut rng = ChaCha20Rng::seed_from_u64(0);
    // with tolerance 0.0001, 4 points gives 792 indices and 300 points gives
    // 89,952
    (0..4)
        .map(|_| {
            let x = rng.gen::<f64>() * 2.0 - 1.0;
            let y = rng.gen::<f64>() * 2.0 - 1.0;
            [x, y]
        })
        .collect()
}

impl Vertex {
    fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Self, position) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Self, color) as u32,
            },
        ]
    }
}

fn create_pipeline<D: DeviceV1_0>(
    device: &D,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    shader_stages: [vk::PipelineShaderStageCreateInfo; 2],
    binding_descriptions: &[vk::VertexInputBindingDescription],
    attribute_descriptions: &[vk::VertexInputAttributeDescription],
) -> vk::Pipeline {
    // vertex format
    let pipeline_vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineVertexInputStateCreateFlags::empty(),
        vertex_binding_description_count: binding_descriptions.len() as u32,
        p_vertex_binding_descriptions: binding_descriptions.as_ptr(),
        vertex_attribute_description_count: attribute_descriptions.len() as u32,
        p_vertex_attribute_descriptions: attribute_descriptions.as_ptr(),
    };

    let pipeline_input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        primitive_restart_enable: vk::FALSE,
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
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A,
    }];

    let viewport_state_info = vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineViewportStateCreateFlags::empty(),
        viewport_count: 1,
        p_viewports: ptr::null(),
        scissor_count: 1,
        p_scissors: ptr::null(),
    };

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

    // dynamic state
    let dynamic_state_flags = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state_info = vk::PipelineDynamicStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineDynamicStateCreateFlags::empty(),
        dynamic_state_count: dynamic_state_flags.len() as u32,
        p_dynamic_states: dynamic_state_flags.as_ptr(),
    };

    let pipeline_infos = [vk::GraphicsPipelineCreateInfo {
        s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineCreateFlags::empty(),
        stage_count: shader_stages.len() as u32,
        p_stages: shader_stages.as_ptr(),
        p_vertex_input_state: &pipeline_vertex_input_info,
        p_input_assembly_state: &pipeline_input_assembly_info,
        p_tessellation_state: ptr::null(),
        p_viewport_state: &viewport_state_info,
        p_rasterization_state: &pipeline_rasterization_info,
        p_multisample_state: &pipeline_multisample_info,
        p_depth_stencil_state: ptr::null(),
        p_color_blend_state: &pipeline_color_blend_info,
        p_dynamic_state: &dynamic_state_info,
        layout: pipeline_layout,
        render_pass,
        subpass: 0,
        base_pipeline_handle: vk::Pipeline::null(),
        base_pipeline_index: 0,
    }];

    unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None) }
        .expect("Couldn't create graphics pipeline")[0]
}

fn create_pipeline_layout<D: DeviceV1_0>(device: &D) -> vk::PipelineLayout {
    // fixed-function pipeline settings

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

    unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .expect("Couldn't create pipeline layout!")
    }
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

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
fn extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        XlibSurface::name().as_ptr(),
        DebugUtils::name().as_ptr(),
    ]
}

#[cfg(target_os = "macos")]
fn extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        MacOSSurface::name().as_ptr(),
        DebugUtils::name().as_ptr(),
    ]
}

#[cfg(all(windows))]
fn extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        Win32Surface::name().as_ptr(),
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

// many of these functions are ripped from https://github.com/bwasty/vulkan-tutorial-rs

// this is ripped from the ash examples
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use winit::os::unix::WindowExt;
    let x11_display = window.get_xlib_display().unwrap();
    let x11_window = window.get_xlib_window().unwrap();
    let x11_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
        .window(x11_window)
        .dpy(x11_display as *mut vk::Display);

    let xlib_surface_loader = XlibSurface::new(entry, instance);
    xlib_surface_loader.create_xlib_surface(&x11_create_info, None)
}

#[cfg(target_os = "macos")]
unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use std::ptr;
    use winit::os::macos::WindowExt;

    let wnd: cocoa_id = mem::transmute(window.get_nswindow());

    let layer = CoreAnimationLayer::new();

    layer.set_edge_antialiasing_mask(0);
    layer.set_presents_with_transaction(false);
    layer.remove_all_animations();

    let view = wnd.contentView();

    layer.set_contents_scale(view.backingScaleFactor());
    view.setLayer(mem::transmute(layer.as_ref()));
    view.setWantsLayer(YES);

    let create_info = vk::MacOSSurfaceCreateInfoMVK {
        s_type: vk::StructureType::MACOS_SURFACE_CREATE_INFO_M,
        p_next: ptr::null(),
        flags: Default::default(),
        p_view: window.get_nsview() as *const c_void,
    };

    let macos_surface_loader = MacOSSurface::new(entry, instance);
    macos_surface_loader.create_mac_os_surface_mvk(&create_info, None)
}

#[cfg(target_os = "windows")]
unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use std::ptr;
    use winapi::shared::windef::HWND;
    use winapi::um::libloaderapi::GetModuleHandleW;
    use winit::os::windows::WindowExt;

    let hwnd = window.get_hwnd() as HWND;
    let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
    let win32_create_info = vk::Win32SurfaceCreateInfoKHR {
        s_type: vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        hinstance,
        hwnd: hwnd as *const c_void,
    };
    let win32_surface_loader = Win32Surface::new(entry, instance);
    win32_surface_loader.create_win32_surface(&win32_create_info, None)
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

fn pixel_to_vk(screen_dims: vk::Extent2D, pixel_pos: &PixelPos) -> VkPos {
    let screen_width = screen_dims.width as f64;
    let screen_height = screen_dims.height as f64;

    [
        (pixel_pos[0] - screen_width * 0.5) / screen_width * 2.0,
        (pixel_pos[1] - screen_height * 0.5) / screen_height * 2.0,
    ]
}

fn vk_to_point(vk_pos: &VkPos) -> lyon::math::Point {
    lyon::math::point(vk_pos[0] as f32, vk_pos[1] as f32)
}

fn pixel_dist(a: &PixelPos, b: &PixelPos) -> f64 {
    ((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])).sqrt()
}
