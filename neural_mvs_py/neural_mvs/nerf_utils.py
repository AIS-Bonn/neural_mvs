#functions taken from https://github.com/krrish94/nerf-pytorch/blob/master/tiny_nerf.py

import math
from typing import Optional

import torch
import torch.nn.functional as F

# import torchsearchsorted


from easypbr  import Profiler
#Just to have something close to the macros we have in c++
def profiler_start(name):
    if(Profiler.is_profiling_gpu()):
        torch.cuda.synchronize()
    Profiler.start(name)
def profiler_end(name):
    if(Profiler.is_profiling_gpu()):
        torch.cuda.synchronize()
    Profiler.end(name)
TIME_START = lambda name: profiler_start(name)
TIME_END = lambda name: profiler_end(name)


#We create it here because otherwise for some reason the GPU synronizes when we create and kinda slows everything down
one_e_10 = torch.tensor([1e10], dtype=torch.float32).to("cuda")



def img2mse(img_src, img_tgt):
    return torch.nn.functional.mse_loss(img_src, img_tgt)


def mse2psnr(mse):
    # For numerical stability, avoid a zero mse loss.
    if mse == 0:
        mse = 1e-5
    return -10.0 * math.log10(mse)


def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i : i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def meshgrid_xy(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)
    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.
    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod


def get_ray_bundle(
    height: int, width: int, fx: float, fy: float, cx: float, cy: float,  tform_cam2world: torch.Tensor, novel=False
):
    r"""Compute the bundle of rays passing through all pixels of an image (one ray per pixel).
    Args:
    height (int): Height of an image (number of pixels).
    width (int): Width of an image (number of pixels).
    focal_length (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
      transforms a 3D point from the camera frame to the "world" frame for the current example.
    Returns:
    ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
      each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
      row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    ray_directions (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
      direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
      passing through the pixel at row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    """
    # TESTED
    ii, jj = meshgrid_xy(
        torch.arange(
            width, dtype=tform_cam2world.dtype, device=tform_cam2world.device
        ).to(tform_cam2world),
        torch.arange(
            height, dtype=tform_cam2world.dtype, device=tform_cam2world.device
        ),
    )
    if novel:
        directions = torch.stack(
            [
                (ii - cx) / fx,
                -(jj - cy) / fy,
                -torch.ones_like(ii),
            ],
            dim=-1,
        )
    else: 
        directions = torch.stack(
            [
                (ii - cx) / fx,
                -(jj - cy) / fy,
                torch.ones_like(ii),
            ],
            dim=-1,
        )
    ray_directions = torch.sum(
        directions[..., None, :] * tform_cam2world[:3, :3], dim=-1
    )
    #ray directions have to be normalized 
    # ray_directions=F.normalize(ray_directions, p=2, dim=2)
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    return ray_origins, ray_directions


def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=1)


def get_embedding_function(
    num_encoding_functions=6, include_input=True, log_sampling=True
):
    r"""Returns a lambda function that internally calls positional_encoding.
    """
    return lambda x: positional_encoding(
        x, num_encoding_functions, include_input, log_sampling
    )


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # UNTESTED, but fairly sure.

    # Shift rays origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = (
        -1.0
        / (W / (2.0 * focal))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = (
        -1.0
        / (H / (2.0 * focal))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def gather_cdf_util(cdf, inds):
    r"""A very contrived way of mimicking a version of the tf.gather()
    call used in the original impl.
    """
    orig_inds_shape = inds.shape
    inds_flat = [inds[i].view(-1) for i in range(inds.shape[0])]
    valid_mask = [
        torch.where(ind >= cdf.shape[1], torch.zeros_like(ind), torch.ones_like(ind))
        for ind in inds_flat
    ]
    inds_flat = [
        torch.where(ind >= cdf.shape[1], (cdf.shape[1] - 1) * torch.ones_like(ind), ind)
        for ind in inds_flat
    ]
    cdf_flat = [cdf[i][ind] for i, ind in enumerate(inds_flat)]
    cdf_flat = [cdf_flat[i] * valid_mask[i] for i in range(len(cdf_flat))]
    cdf_flat = [
        cdf_chunk.reshape([1] + list(orig_inds_shape[1:])) for cdf_chunk in cdf_flat
    ]
    return torch.cat(cdf_flat, dim=0)


def sample_pdf(bins, weights, num_samples, det=False):
    # TESTED (Carefully, line-to-line).
    # But chances of bugs persist; haven't integration-tested with
    # training routines.

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / weights.sum(-1).unsqueeze(-1)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf), -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, num_samples).to(weights)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples]).to(weights)

    # Invert CDF
    inds = torchsearchsorted.searchsorted(
        cdf.contiguous(), u.contiguous(), side="right"
    )
    below = torch.max(torch.zeros_like(inds), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), -1)
    orig_inds_shape = inds_g.shape

    cdf_g = gather_cdf_util(cdf, inds_g)
    bins_g = gather_cdf_util(bins, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def sample_pdf_2(bins, weights, num_samples, det=False):
    r"""sample_pdf function from another concurrent pytorch implementation
    by yenchenlin (https://github.com/yenchenlin/nerf-pytorch).
    """

    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
    )  # (batchsize, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(
            0.0, 1.0, steps=num_samples, dtype=weights.dtype, device=weights.device
        )
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.rand(
            list(cdf.shape[:-1]) + [num_samples],
            dtype=weights.dtype,
            device=weights.device,
        )

    # Invert CDF
    u = u.contiguous()
    cdf = cdf.contiguous()
    inds = torchsearchsorted.searchsorted(cdf, u, side="right")
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), dim=-1)  # (batchsize, num_samples, 2)

    matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def compute_query_points_from_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    num_samples: int,
    randomize: Optional[bool] = True,
) -> (torch.Tensor, torch.Tensor):
    r"""Compute query 3D points given the "bundle" of rays. The near_thresh and far_thresh
    variables indicate the bounds within which 3D points are to be sampled.
    Args:
        ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
          `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
        ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
          `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
        near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
          coordinate that is of interest/relevance).
        far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
          coordinate that is of interest/relevance).
        num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
          randomly, whilst trying to ensure "some form of" uniform spacing among them.
        randomize (optional, bool): Whether or not to randomize the sampling of query points.
          By default, this is set to `True`. If disabled (by setting to `False`), we sample
          uniformly spaced points along each ray in the "bundle".
    Returns:
        query_points (torch.Tensor): Query points along each ray
          (shape: :math:`(width, height, num_samples, 3)`).
        depth_values (torch.Tensor): Sampled depth values along each ray. This is the euclidean distance between the query point and the sensor origin
          (shape: :math:`(num_samples)`) or :math:`(width, height, 3)`) in the case you use randomize=True.
    """
    # TESTED
    # shape: (num_samples)
    depth_values = torch.linspace(near_thresh, far_thresh, num_samples).to(ray_origins)
    # print("depth_values after linspace is ", depth_values.shape)
    if randomize is True:
        # ray_origins: (width, height, 3)
        # noise_shape = (width, height, num_samples)
        noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
        # depth_values: (num_samples)
        depth_values = (
            depth_values
            + torch.rand(noise_shape).to(ray_origins)
            * (far_thresh - near_thresh)
            / num_samples
        )
    # (width, height, num_samples, 3) = (width, height, 1, 3) + (width, height, 1, 3) * (num_samples, 1)
    # query_points:  (width, height, num_samples, 3)
    query_points = (
        ray_origins[..., None, :]
        + ray_directions[..., None, :] * depth_values[..., :, None]
    )
    # TODO: Double-check that `depth_values` returned is of shape `(num_samples)`.
    return query_points, depth_values


#takes near and far as tensors
def compute_query_points_from_rays2(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_thresh: torch.Tensor,
    far_thresh: torch.Tensor,
    num_samples: int,
    randomize: Optional[bool] = True,
) -> (torch.Tensor, torch.Tensor):
    r"""Compute query 3D points given the "bundle" of rays. The near_thresh and far_thresh
    variables indicate the bounds within which 3D points are to be sampled.
    Args:
        ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
          `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
        ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
          `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
        near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
          coordinate that is of interest/relevance).
        far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
          coordinate that is of interest/relevance).
        num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
          randomly, whilst trying to ensure "some form of" uniform spacing among them.
        randomize (optional, bool): Whether or not to randomize the sampling of query points.
          By default, this is set to `True`. If disabled (by setting to `False`), we sample
          uniformly spaced points along each ray in the "bundle".
    Returns:
        query_points (torch.Tensor): Query points along each ray
          (shape: :math:`(width, height, num_samples, 3)`).
        depth_values (torch.Tensor): Sampled depth values along each ray. This is the euclidean distance between the query point and the sensor origin
          (shape: :math:`(num_samples)`) or :math:`(width, height, 3)`) in the case you use randomize=True.
    """
    # TESTED
    # shape: (num_samples)
    depth_values = torch.linspace(0.0, 1.0, num_samples).to(ray_origins) #just a vector of size [num_samples]

    height=ray_origins.shape[0]
    width=ray_origins.shape[1]

    depth_values_img=depth_values.view(1,1,-1).repeat(height,width,1) #make it [height,width,num_samples]
    # print("ray_origin.shape", ray_origins.shape)
    # print("depth_values_img has shape ", depth_values_img.shape )

    noise=torch.rand([height,width,num_samples]).to(ray_origins)
    # print("noise has shape ", noise.shape)
    # print("near is ", near_thresh)

    #the depth_values_img need to be not in range 0, 1 but in range near-far for every pixel
    range_of_samples_img=far_thresh-near_thresh #says for each pixel what is the range of values that the sampled points will take
    range_of_samples_img=range_of_samples_img.view(height,width,1)
    # print("range_of_samples_img is ", range_of_samples_img.shape )
    depth_values_img=depth_values_img*range_of_samples_img+near_thresh.view(height,width,1)


    #get the noise instead of from [0,1] range to a range of [-sample_room,+sample_room]
    #each sample has a certain wigglle room in the negative and positive direction before it collides with the other samples
    wiggle_room_img=range_of_samples_img/num_samples
    noise=noise*wiggle_room_img

    #apply noiset o depth values
    depth_values_img=depth_values_img+noise

    # query_points=ray_origins + ray_directions*depth_values_img
    query_points = (
        ray_origins[..., None, :]
        + ray_directions[..., None, :] * depth_values_img[..., :, None] )

    return query_points, depth_values_img



    # #create a tensor of depth values which has shape  witdh,height, num_samples with values linsapced between 0 and 1
    # print("depth valus has shape ", depth_values.shape )


    # # print("depth_values after linspace is ", depth_values.shape)
    # if randomize is True:
    #     # ray_origins: (width, height, 3)
    #     # noise_shape = (width, height, num_samples)
    #     noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
    #     # depth_values: (num_samples)
    #     depth_values = (
    #         depth_values
    #         + torch.rand(noise_shape).to(ray_origins)
    #         * (far_thresh - near_thresh)
    #         / num_samples
    #     )
    # # (width, height, num_samples, 3) = (width, height, 1, 3) + (width, height, 1, 3) * (num_samples, 1)
    # # query_points:  (width, height, num_samples, 3)
    # query_points = (
    #     ray_origins[..., None, :]
    #     + ray_directions[..., None, :] * depth_values[..., :, None]
    # )
    # # TODO: Double-check that `depth_values` returned is of shape `(num_samples)`.
    # return query_points, depth_values



def render_volume_density(
    radiance_field: torch.Tensor, ray_origins: torch.Tensor, depth_values: torch.Tensor, siren_out_channels
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    r"""Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.
    Args:
    radiance_field (torch.Tensor): A "field" where, at each query location (X, Y, Z),
      we have an emitted (RGB) color and a volume density (denoted :math:`\sigma` in
      the paper) (shape: :math:`(width, height, num_samples, 4)`).
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
      (shape: :math:`(num_samples)`).
    Returns:
    rgb_map (torch.Tensor): Rendered RGB image (shape: :math:`(width, height, 3)`).
    depth_map (torch.Tensor): Rendered depth image (shape: :math:`(width, height)`).
    acc_map (torch.Tensor): # TODO: Double-check (I think this is the accumulated
      transmittance map).
    """

    # TESTED
    # sigma_a = torch.relu(radiance_field[..., siren_out_channels-1])
    # rgb = torch.sigmoid(radiance_field[..., :siren_out_channels-1])
    # TIME_START("why")
    sigma_a = radiance_field[..., siren_out_channels-1]
    rgb = radiance_field[..., :siren_out_channels-1]
    # one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device) #we dont create it here because for some reason teh gpu syncronizes here and stalls the pipeline
    # TIME_END("why")
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    alpha = 1.0 - torch.exp(-sigma_a * dists) #as explained in willians: A volume density optical model. This normalizes the value between 0 and 1
    # print("alpha has range ", alpha.min().item(), " ", alpha.max().item() )
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    weights_sum=weights.sum(dim=-1)
    weights_sum=weights_sum+0.00001
    # print("weight_sum", weights_sum.shape)

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)   
    # depth_map = (weights * depth_values).sum(dim=-1) 
    depth_map = (weights * depth_values).sum(dim=-1) / weights_sum #normalize so that we get actual depth
    acc_map = weights.sum(-1)

    # print("rgb map ", rgb_map.shape)
    # print("depth_map map ", depth_map.shape)

    return rgb_map, depth_map, acc_map


#trying to make my own renderer based on nerfplusplus because the nerf default one creates lots of splodges of color in empty space 
#based on https://github.com/Kai-46/nerfplusplus/blob/b24b9047ade68166c1a9792554e2aef60dd137cc/ddp_model.py
def render_volume_density_nerfplusplus(
    radiance_field: torch.Tensor, ray_origins: torch.Tensor, depth_values: torch.Tensor, siren_out_channels
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    r"""Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.
    Args:
    radiance_field (torch.Tensor): A "field" where, at each query location (X, Y, Z),
      we have an emitted (RGB) color and a volume density (denoted :math:`\sigma` in
      the paper) (shape: :math:`(width, height, num_samples, 4)`).
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
      (shape: :math:`(num_samples)`).
    Returns:
    rgb_map (torch.Tensor): Rendered RGB image (shape: :math:`(width, height, 3)`).
    depth_map (torch.Tensor): Rendered depth image (shape: :math:`(width, height)`).
    acc_map (torch.Tensor): # TODO: Double-check (I think this is the accumulated
      transmittance map).
    """
    # TESTED
    # sigma_a = torch.abs(radiance_field[..., 3])
    # sigma_a = torch.sigmoid(radiance_field[..., 3])
    # rgb = torch.sigmoid(radiance_field[..., :3])
    sigma_a = radiance_field[..., siren_out_channels-1]
    rgb = radiance_field[..., :siren_out_channels-1]

    # fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]

    # # alpha blending
    # fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
    # # account for view directions
    # fg_dists = ray_d_norm * torch.cat((fg_dists, fg_z_max.unsqueeze(-1) - fg_z_vals[..., -1:]), dim=-1)  # [..., N_samples]
    # fg_alpha = 1. - torch.exp(-fg_raw['sigma'] * fg_dists)  # [..., N_samples]
    # T = torch.cumprod(1. - fg_alpha + TINY_NUMBER, dim=-1)   # [..., N_samples]
    # bg_lambda = T[..., -1]
    # T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]
    # fg_weights = fg_alpha * T     # [..., N_samples]
    # fg_rgb_map = torch.sum(fg_weights.unsqueeze(-1) * fg_raw['rgb'], dim=-2)  # [..., 3]
    # fg_depth_map = torch.sum(fg_weights * fg_z_vals, dim=-1) # [...,]
    # print("depth_values has shape", depth_values.shape)
    
    # one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
    # # print("depth values when concating has shape ", depth_values[..., :1].shape)
    # # print("last depth values is ", depth_values[..., :1])

    # # print("dv 1 shape:", depth_values[..., 1:].shape)
    # # print("dv :-1 shape ", depth_values[..., -1].shape)
    # # print("dv 1:", depth_values[..., 1:])
    # # print("dv :-1", depth_values[..., -1])
    # # dist_calc =depth_values[..., 1:] - depth_values[..., :-1]
    # # print("dist_calc by substraction", dist_calc)

    # dists = torch.cat(
    #     (
    #         depth_values[..., 1:] - depth_values[..., :-1],
    #         one_e_10.expand(depth_values[..., :1].shape),
    #     ),
    #     dim=-1,
    # )
    # # print("dist is ", dists)
    # # alpha = 1.0 - torch.exp(-sigma_a * dists)
    # alpha = 1.0 - sigma_a * dists
    # # alpha=sigma_a
    # weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    # rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    # # print("rgb map maximum is ", rgb_map.min(), " ", rgb_map.max() )
    # depth_map = (weights * depth_values).sum(dim=-1)
    # acc_map = weights.sum(-1)





    #my own 
    # nr_samples_per_ray=100
    # sigma_a=sigma_a* (1.0/nr_samples_per_ray)
    # sigma_cum=torch.cumsum(sigma_a, dim=2)
    # sigma_cum_above_1 = sigma_cum>1.0
    # sigma_cum[sigma_cum_above_1] = 0
    # # print("sigma_cum", sigma_cum)
    # # exit(1)

    # sigma_cum_total_weight=sigma_cum.sum(2, keepdim=True)

    # rgb_map = (sigma_cum[..., None] * rgb).sum(dim=-2)
    # depth_map = (sigma_cum * depth_values).sum(dim=-1)
    # acc_map = sigma_cum.sum(-1)

    # # print("rgb map maximum is ", rgb_map.min(), " ", rgb_map.max() )
    # rgb_map=rgb_map/sigma_cum_total_weight
    # depth_map=depth_map/sigma_cum_total_weight.squeeze(2)





    #based on  https://developer.nvidia.com/sites/all/modules/custom/gpugems/books/GPUGems/gpugems_ch39.html
    # nr_samples_per_ray=30
    # sigma_a=sigma_a* (1.0/nr_samples_per_ray)
    # sigma_cum=torch.cumprod(1-sigma_a, dim=2)
    sigma_cum=cumprod_exclusive(1-sigma_a)
    sigma_cum_above_1 = sigma_cum>1.0
    sigma_cum[sigma_cum_above_1] = 0
    # print("sigma_cum", sigma_cum)
    # exit(1)

    sigma_cum_total_weight=sigma_cum.sum(2, keepdim=True)

    rgb_map = (sigma_cum[..., None] * rgb).sum(dim=-2)
    depth_map = (sigma_cum * depth_values).sum(dim=-1)
    acc_map = sigma_cum.sum(-1)

    # print("rgb map maximum is ", rgb_map.min(), " ", rgb_map.max() )
    # rgb_map=rgb_map/sigma_cum_total_weight
    # depth_map=depth_map/sigma_cum_total_weight.squeeze(2)



    return rgb_map, depth_map, acc_map


#just trying out soem random stuff
def render_volume_density2(
    radiance_field: torch.Tensor, ray_origins: torch.Tensor, depth_values: torch.Tensor, siren_out_channels
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    r"""Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.
    Args:
    radiance_field (torch.Tensor): A "field" where, at each query location (X, Y, Z),
      we have an emitted (RGB) color and a volume density (denoted :math:`\sigma` in
      the paper) (shape: :math:`(width, height, num_samples, 4)`).
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
      (shape: :math:`(num_samples)`).
    Returns:
    rgb_map (torch.Tensor): Rendered RGB image (shape: :math:`(width, height, 3)`).
    depth_map (torch.Tensor): Rendered depth image (shape: :math:`(width, height)`).
    acc_map (torch.Tensor): # TODO: Double-check (I think this is the accumulated
      transmittance map).
    """

    # TESTED
    # sigma_a = torch.relu(radiance_field[..., siren_out_channels-1])
    # rgb = torch.sigmoid(radiance_field[..., :siren_out_channels-1])
    sigma_a = radiance_field[..., siren_out_channels-1]
    rgb = radiance_field[..., :siren_out_channels-1]
    one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    # alpha = sigma_a * dists
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    weights_sum=weights.sum(dim=-1)
    weights_sum=weights_sum+0.00001
    # print("weight_sum", weights_sum.shape)

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)  
    # depth_map = (weights * depth_values).sum(dim=-1) 
    depth_map = (weights * depth_values).sum(dim=-1) / weights_sum #normalize so that we get actual depth
    acc_map = weights.sum(-1)

    # print("rgb map ", rgb_map.shape)
    # print("depth_map map ", depth_map.shape)

    return rgb_map, depth_map, acc_map



