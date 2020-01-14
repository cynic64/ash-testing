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

const SC_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

pub fn main() {
    // create winit window
    let mut events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new()
        .with_title("Ash - Example")
        .build(&events_loop)
        .unwrap();

    // create instance
    let app_info = vk::ApplicationInfo {
        api_version: vk_make_version!(1, 1, 0),
        ..Default::default()
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

    let mut debug_utils_create_info = vk::DebugUtilsMessengerCreateInfoEXT {
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

    let create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_layer_names(&layers_names_raw)
        .enabled_extension_names(&extension_names_raw)
        .push_next(&mut debug_utils_create_info)
        .build();

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
    let device_queue_create_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .queue_priorities(&[1.0])
        .build();

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&[device_queue_create_info])
        .enabled_extension_names(&get_device_extensions_raw())
        .build();

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

    // create swapchain
    let sc_format = vk::SurfaceFormatKHR {
        format: SC_FORMAT,
        color_space: vk::ColorSpaceKHR::default(),
    };

    let sc_present_mode = vk::PresentModeKHR::IMMEDIATE;

    let sc_create_info = vk::SwapchainCreateInfoKHR::builder()
        .min_image_count(3)
        .image_format(vk::Format::B8G8R8A8_UNORM)
        .present_mode(vk::PresentModeKHR::IMMEDIATE)
        .surface(surface)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .image_array_layers(1)
        .image_extent(starting_dims)
        .build();

    let swapchain_creator = Swapchain::new(&instance, &device);
    let swapchain = unsafe { swapchain_creator.create_swapchain(&sc_create_info, None) }
        .expect("Couldn't create swapchain");

    let images = unsafe { swapchain_creator.get_swapchain_images(swapchain) }
        .expect("Couldn't get swapchain images");

    let image_views: Vec<_> = images
        .iter()
        .map(|image| {
            let iv_info = vk::ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(SC_FORMAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
        })
        .collect();

    // shaders
    let frag_code = read_shader_code(&relative_path("shaders/vt-3/triangle.frag.spv"));
    let vert_code = read_shader_code(&relative_path("shaders/vt-3/triangle.vert.spv"));

    let frag_module = create_shader_module(&device, frag_code);
    let vert_module = create_shader_module(&device, vert_code);

    let vert_stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_module)
        .name(&CString::new("main").unwrap())
        .build();

    let frag_stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_module)
        .name(&CString::new("main").unwrap())
        .build();

    let shader_stages = [vert_stage_info, frag_stage_info];

    // fixed-function pipeline settings

    // a.k.a vertex format
    // we don't reallt have a format since they are hard-coded into the vertex
    // shader for now
    let pipeline_vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder().build();

    let pipeline_input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .build();

    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: starting_dims.width as f32,
        height: starting_dims.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };

    let scissors = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: starting_dims,
    };

    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&[viewport])
        .scissors(&[scissors])
        .build();

    let pipeline_rasterization_info = vk::PipelineRasterizationStateCreateInfo::builder()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .build();

    let pipeline_multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .build();

    // color blending info per framebuffer
    let pipeline_color_blend_attachment_info = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .blend_enable(false)
        .build();

    // color blending settings for the whole pipleine
    let pipeline_color_blend_info = vk::PipelineColorBlendStateCreateInfo::builder()
        .attachments(&[pipeline_color_blend_attachment_info])
        .build();

    // we don't use any shader uniforms so we can leave it empty
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder().build();

    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .expect("Couldn't create pipeline layout!")
    };

    // render pass
    let render_pass = create_render_pass(&device);

    // shader modules only need to live long enough to create the pipeline
    unsafe {
        device.destroy_shader_module(frag_module, None);
        device.destroy_shader_module(vert_module, None);
    }

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
    }

    // destroy objects
    unsafe {
        device.destroy_pipeline_layout(pipeline_layout, None);
        swapchain_creator.destroy_swapchain(swapchain, None);
        device.destroy_render_pass(render_pass, None);
        device.destroy_device(None);
        surface_loader.destroy_surface(surface, None);
        debug_utils_loader.destroy_debug_utils_messenger(debug_utils_messenger, None);
        instance.destroy_instance(None);
    }
}

fn create_shader_module<D: DeviceV1_0>(device: &D, code: Vec<u8>) -> vk::ShaderModule {
    use ash::util::read_spv;
    use std::io::Cursor;

    let readable_code = read_spv(&mut Cursor::new(&code)).expect("Couldn't read SPV");
    let shader_module_create_info = vk::ShaderModuleCreateInfo::builder()
        .code(&readable_code)
        .build();

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
    let attachment_desc = vk::AttachmentDescription::builder()
        .format(SC_FORMAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .build();


    let attachment_ref = vk::AttachmentReference::builder()
    // the previous attachment will be the 0th attachment in the attachment
    // array, which we will construct later
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build();

    let color_attachment_refs = [attachment_ref];
    let subpass_desc = vk::SubpassDescription::builder()
    // the indices correspond to what you'd write for the output layout in
    // the fragment shader stage. To output to this image, for example,
    // you'd write `layout(location = 0) out vec4 f_color`
        .color_attachments(&color_attachment_refs)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .build();

    let render_pass_info = vk::RenderPassCreateInfo::builder()
    // the previously mentioned attachment array
        // .attachments(&render_pass_attachments)
        .attachments(&[attachment_desc])
        .subpasses(&[subpass_desc])
        .build();

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
