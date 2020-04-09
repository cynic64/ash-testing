use ash::extensions::khr;
use ash::vk;
use ash::{Device, Instance};
use ash::version::DeviceV1_0;

use std::ptr;

const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

pub struct Vindow {
    // taken
    physical_device: vk::PhysicalDevice,
    device: Device,
    queue: vk::Queue,

    render_pass: vk::RenderPass,

    surface_loader: khr::Surface,
    surface: vk::SurfaceKHR,

    // created
    swapchain_creator: khr::Swapchain,
    vol: Option<Volatile>,

    acquired_idx: Option<u32>,
}

// Components of the swapchain that will be nonexistent during recreation.
// The point of this is that I can .take() all the components that need to be
// destroyed during swapchain recreation, and therefore make sure noone tries
// to access the briefly-void vulkan pointers.

struct Volatile {
    swapchain: vk::SwapchainKHR,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_dims: vk::Extent2D,
    framebuffers: Vec<vk::Framebuffer>,
}

impl Vindow {
    pub fn new(
        physical_device: vk::PhysicalDevice,
        device: Device,
        instance: Instance,
        queue: vk::Queue,
        render_pass: vk::RenderPass,
        surface_loader: khr::Surface,
        surface: vk::SurfaceKHR,
    ) -> Self {
        let swapchain_creator = khr::Swapchain::new(&instance, &device);

        let (swapchain, swapchain_images, swapchain_dims) = create_swapchain(
            physical_device,
            &surface_loader,
            surface,
            &swapchain_creator,
        );

        let swapchain_image_views = create_swapchain_image_views(&device, &swapchain_images);

        let framebuffers = create_framebuffers(&device, render_pass, swapchain_dims, &swapchain_image_views);

        Self {
            physical_device,
            device,
            queue,

            render_pass,

            surface_loader,
            surface,

            swapchain_creator,
            vol: Some(Volatile {
                swapchain,
                swapchain_image_views,
                swapchain_dims,
                framebuffers,
            }),

            acquired_idx: None,
        }
    }

    pub fn acquire(&mut self, semaphore: vk::Semaphore) -> (vk::Framebuffer, vk::Extent2D) {
        // <Semaphore> will be signalled when the image is available. Make sure
        // isn't already in use by a different frame in flight.

        // Returns the (image index, framebuffer to use, swapchain dimensions)

        let acquire_result = unsafe {
            self.swapchain_creator.acquire_next_image(
                self.vol.as_ref().unwrap().swapchain,
                std::u64::MAX,
                semaphore,
                vk::Fence::null(),
            )
        };

        let image_index = match acquire_result {
            Ok((image_idx, _is_sub_optimal)) => image_idx,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate();
                self.acquire(semaphore);
                self.acquired_idx.unwrap()
            },
            Err(e) => panic!("Unexpected error during acquire_next_image: {}", e),
        };

        self.acquired_idx = Some(image_index);

        (self.vol.as_ref().unwrap().framebuffers[image_index as usize], self.vol.as_ref().unwrap().swapchain_dims)
    }

    pub fn present(&mut self, semaphore: vk::Semaphore) {
        let wait_semaphores = [semaphore];
        let swapchains = [self.vol.as_ref().unwrap().swapchain];
        let image_indices = [self.acquired_idx.expect("No acquired image index found when trying to present! Did you forget to acquire before presentation?")];

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: image_indices.as_ptr(),
            p_results: ptr::null_mut(),
        };

        match unsafe {
            self.swapchain_creator
                .queue_present(self.queue, &present_info)
        } {
            Ok(_idk_what_this_is) => {},
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate();
            },
            Err(e) => panic!("Unexpected error during queue_present: {}", e),
        }
    }

    pub fn cleanup(mut self) {
        let vol = self.vol.take().expect("Volatile swapchain objects not present when trying to clean up");

        unsafe {
            vol.swapchain_image_views
                .iter()
                .for_each(|&view| self.device.destroy_image_view(view, None));

            vol.framebuffers
                .iter()
                .for_each(|&fb| self.device.destroy_framebuffer(fb, None));

            self.swapchain_creator.destroy_swapchain(vol.swapchain, None);
        }
    }

    fn recreate(&mut self) {
        let vol = self.vol.take().unwrap();
        cleanup_swapchain(
            &self.device,
            &self.swapchain_creator,
            vol.swapchain,
            vol.framebuffers,
            vol.swapchain_image_views,
        );

        let (swapchain, swapchain_images, swapchain_dims) = create_swapchain(
            self.physical_device,
            &self.surface_loader,
            self.surface,
            &self.swapchain_creator,
        );

        let swapchain_image_views = create_swapchain_image_views(&self.device, &swapchain_images);

        let framebuffers = create_framebuffers(&self.device, self.render_pass, swapchain_dims, &swapchain_image_views);

        self.vol = Some(Volatile {
            swapchain,
            swapchain_image_views,
            swapchain_dims,
            framebuffers
        });
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

fn create_swapchain(
    physical_device: vk::PhysicalDevice,
    surface_loader: &khr::Surface,
    surface: vk::SurfaceKHR,
    swapchain_creator: &khr::Swapchain,
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

fn check_device_swapchain_caps(
    surface_loader: &khr::Surface,
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

fn cleanup_swapchain(
    device: &Device,
    swapchain_creator: &khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    framebuffers: Vec<vk::Framebuffer>,
    swapchain_image_views: Vec<vk::ImageView>,
) {
    unsafe {
        device
            .device_wait_idle()
            .expect("Couldn't wait on device idle")
    };

    // destroy framebuffers
    framebuffers
        .iter()
        .for_each(|&fb| unsafe { device.destroy_framebuffer(fb, None) });

    // destroy swapchain image views
    swapchain_image_views
        .iter()
        .for_each(|&view| unsafe { device.destroy_image_view(view, None) });

    // destroy swapchain
    unsafe { swapchain_creator.destroy_swapchain(swapchain, None) };
}

