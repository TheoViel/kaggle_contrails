import numpy as np
import gunpowder as gp
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import convolve, gaussian_filter


def get_shapedescript(
    segmentation,
    sigma,
    components=None,
    voxel_size=None,
    roi=None,
    labels=None,
    mode="gaussian",
    downsample=1,
):
    return ShapeDescript(sigma, mode, downsample).get_descriptors(
        segmentation, components, voxel_size, roi, labels
    )


class ShapeDescript(object):
    def __init__(self, sigma, mode="gaussian", downsample=1):
        self.sigma = sigma
        self.mode = mode
        self.downsample = downsample
        self.coords = {}

    def get_descriptors(
        self, segmentation, components=None, voxel_size=None, roi=None, labels=None
    ):
        dims = len(segmentation.shape)

        if voxel_size is None:
            voxel_size = gp.Coordinate((1,) * dims)
        else:
            voxel_size = gp.Coordinate(voxel_size)

        if roi is None:
            roi = gp.Roi((0,) * dims, segmentation.shape)

        roi_slices = roi.to_slices()

        if labels is None:
            labels = np.unique(segmentation[roi_slices])

        # get number of channels
        if components is None:
            if dims == 2:
                self.sigma = self.sigma[0:2]
                channels = 6
            elif dims == 3:
                channels = 10
            else:
                raise AssertionError(f"Segmentation shape has {dims} dims.")

        else:
            channels = len(components)

        # prepare full-res descriptor volumes for roi
        descriptors = np.zeros((channels,) + roi.get_shape(), dtype=np.float32)

        # get sub-sampled shape, roi, voxel size and sigma
        df = self.downsample

        sub_shape = tuple(s / df for s in segmentation.shape)
        sub_roi = roi / df

        assert sub_roi * df == roi, (
            "Segmentation shape %s is not a multiple of downsampling factor "
            "%d (sub_roi=%s, roi=%s)."
            % (segmentation.shape, self.downsample, sub_roi, roi)
        )
        sub_voxel_size = tuple(v * df for v in voxel_size)
        sub_sigma_voxel = tuple(s / v for s, v in zip(self.sigma, sub_voxel_size))

        # prepare coords volume (reuse if we already have one)
        if (sub_shape, sub_voxel_size) not in self.coords:
            try:
                # 3d by default
                grid = np.meshgrid(
                    np.arange(0, sub_shape[0] * sub_voxel_size[0], sub_voxel_size[0]),
                    np.arange(0, sub_shape[1] * sub_voxel_size[1], sub_voxel_size[1]),
                    np.arange(0, sub_shape[2] * sub_voxel_size[2], sub_voxel_size[2]),
                    indexing="ij",
                )

            except Exception:
                grid = np.meshgrid(
                    np.arange(0, sub_shape[0] * sub_voxel_size[0], sub_voxel_size[0]),
                    np.arange(0, sub_shape[1] * sub_voxel_size[1], sub_voxel_size[1]),
                    indexing="ij",
                )

            self.coords[(sub_shape, sub_voxel_size)] = np.array(grid, dtype=np.float32)

        coords = self.coords[(sub_shape, sub_voxel_size)]

        # for all labels
        for label in labels:
            if label == 0:
                continue

            mask = (segmentation == label).astype(np.float32)

            try:
                # 3d by default
                sub_mask = mask[::df, ::df, ::df]

            except Exception:
                sub_mask = mask[::df, ::df]

            sub_descriptor = np.concatenate(
                self.__get_stats(coords, sub_mask, sub_sigma_voxel, sub_roi, components)
            )

            descriptor = self.__upsample(sub_descriptor, df)
            descriptors += descriptor * mask[roi_slices]

        # normalize stats

        # get max possible mean offset for normalization
        if self.mode == "gaussian":
            # farthest voxel in context is 3*sigma away, but due to Gaussian
            # weighting, sigma itself is probably a better upper bound
            max_distance = np.array([s for s in self.sigma], dtype=np.float32)
        elif self.mode == "sphere":
            # farthest voxel in context is sigma away, but this is almost
            # impossible to reach as offset -- let's take half sigma
            max_distance = np.array([0.5 * s for s in self.sigma], dtype=np.float32)

        if dims == 3:
            # mean offsets (z,y,x) = [0,1,2]
            # covariance (zz,yy,xx) = [3,4,5]
            # pearsons (zy,zx,yx) = [6,7,8]
            # size = [9]

            if components is None:
                # mean offsets in [0, 1]
                descriptors[[0, 1, 2]] = (
                    descriptors[[0, 1, 2]] / max_distance[:, None, None, None] * 0.5
                    + 0.5
                )
                # pearsons in [0, 1]
                descriptors[[6, 7, 8]] = descriptors[[6, 7, 8]] * 0.5 + 0.5
                # reset background to 0
                descriptors[[0, 1, 2, 6, 7, 8]] *= segmentation[roi_slices] != 0

            else:
                for i, c in enumerate(components):
                    c = int(c)

                    if c in range(0, 3):
                        descriptors[[i]] = (
                            descriptors[[i]] / max_distance[c, None, None, None] * 0.5
                            + 0.5
                        )
                        descriptors[[i]] *= segmentation[roi_slices] != 0

                    elif c in range(6, 9):
                        descriptors[[i]] = descriptors[[i]] * 0.5 + 0.5
                        descriptors[[i]] *= segmentation[roi_slices] != 0

                    else:
                        pass

        else:
            # mean offsets (y,x) = [0,1]
            # covariance (yy,xx) = [2,3]
            # pearsons (yx) = [4]
            # size = [5]

            if components is None:
                # mean offsets in [0, 1]
                descriptors[[0, 1]] = (
                    descriptors[[0, 1]] / max_distance[:, None, None] * 0.5 + 0.5
                )
                # pearsons in [0, 1]
                descriptors[[4]] = descriptors[[4]] * 0.5 + 0.5
                # reset background to 0
                descriptors[[0, 1, 4]] *= segmentation[roi_slices] != 0

            else:
                for i, c in enumerate(components):
                    c = int(c)

                    if c in range(0, 2):
                        descriptors[[i]] = (
                            descriptors[[i]] / max_distance[c, None, None] * 0.5 + 0.5
                        )
                        descriptors[[i]] *= segmentation[roi_slices] != 0

                    elif c == 4:
                        descriptors[[i]] = descriptors[[i]] * 0.5 + 0.5
                        descriptors[[i]] *= segmentation[roi_slices] != 0

        # clip outliers
        np.clip(descriptors, 0.0, 1.0, out=descriptors)

        return descriptors

    def __get_stats(self, coords, mask, sigma_voxel, roi, components):
        # mask for object
        masked_coords = coords * mask

        # number of inside voxels
        count = self.__aggregate(mask, sigma_voxel, self.mode, roi)

        count_len = len(count.shape)

        # avoid division by zero
        count[count == 0] = 1

        # mean

        mean = np.array(
            [
                self.__aggregate(masked_coords[d], sigma_voxel, self.mode, roi)
                for d in range(count_len)
            ]
        )

        mean /= count

        if components is not None:
            calc_mean_offset = True in [
                str(comp) in components for comp in range(count_len)
            ]
            calc_covariance = True in [
                str(comp) in components for comp in range(count_len, 4 * count_len - 3)
            ]

        if components is None or calc_mean_offset:
            mean_offset = mean - coords[(slice(None),) + roi.to_slices()]

        # covariance
        if components is None or calc_covariance:
            coords_outer = self.__outer_product(masked_coords)

            # remove duplicate entries in covariance
            entries = [0, 4, 8, 1, 2, 5] if count_len == 3 else [0, 3, 1]

            covariance = np.array(
                [
                    self.__aggregate(coords_outer[d], sigma_voxel, self.mode, roi)
                    # 3d:
                    # 0 1 2
                    # 3 4 5
                    # 6 7 8
                    # 2d:
                    # 0 1
                    # 2 3
                    for d in entries
                ]
            )

            covariance /= count
            covariance -= self.__outer_product(mean)[entries]

            if count_len == 3:
                # variances of z, y, x coordinates
                variance = covariance[[0, 1, 2]]

                # Pearson coefficients of zy, zx, yx
                pearson = covariance[[3, 4, 5]]

                # normalize Pearson correlation coefficient
                variance[variance < 1e-3] = 1e-3  # numerical stability
                pearson[0] /= np.sqrt(variance[0] * variance[1])
                pearson[1] /= np.sqrt(variance[0] * variance[2])
                pearson[2] /= np.sqrt(variance[1] * variance[2])

                # normalize variances to interval [0, 1]
                variance[0] /= self.sigma[0] ** 2
                variance[1] /= self.sigma[1] ** 2
                variance[2] /= self.sigma[2] ** 2

            else:
                # variances of y, x coordinates
                variance = covariance[[0, 1]]

                # Pearson coefficients of yx
                pearson = covariance[[2]]

                # normalize Pearson correlation coefficient
                variance[variance < 1e-3] = 1e-3  # numerical stability
                pearson /= np.sqrt(variance[0] * variance[1])

                # normalize variances to interval [0, 1]
                variance[0] /= self.sigma[0] ** 2
                variance[1] /= self.sigma[1] ** 2

        if components is not None:
            ret = tuple()

            for i in components:
                i = int(i)

                if count_len == 3:
                    if i in range(0, 3):
                        ret += (mean_offset[[i]],)
                    elif i in range(3, 6):
                        ret += (variance[[i - 3]],)
                    elif i in range(6, 9):
                        ret += (pearson[[i - 6]],)
                    elif i == 9:
                        ret += (count[None, :],)
                    else:
                        raise AssertionError(
                            f"3D lsds have components in range(0,10), encountered {i}"
                        )

                elif count_len == 2:
                    if i in range(0, 2):
                        ret += (mean_offset[[i]],)
                    elif i in range(2, 4):
                        ret += (variance[[i - 2]],)
                    elif i == 4:
                        ret += (pearson,)
                    elif i == 5:
                        ret += (count[None, :],)
                    else:
                        raise AssertionError(
                            f"2D lsds have components in range(0,6), encountered {i}"
                        )

                else:
                    raise AssertionError(f"Number of dims was found to be {count_len}")

        else:
            ret = (mean_offset, variance, pearson, count[None, :])

        return ret

    def __make_sphere(self, radius):
        r2 = np.arange(-radius, radius) ** 2
        dist2 = r2[:, None, None] + r2[:, None] + r2
        return (dist2 <= radius**2).astype(np.float32)

    def __aggregate(self, array, sigma, mode="gaussian", roi=None):
        if roi is None:
            roi_slices = (slice(None),)
        else:
            roi_slices = roi.to_slices()

        if mode == "gaussian":
            return gaussian_filter(
                array, sigma=sigma, mode="constant", cval=0.0, truncate=3.0
            )[roi_slices]

        elif mode == "sphere":
            radius = sigma[0]
            for d in range(len(sigma)):
                assert (
                    radius == sigma[d]
                ), "For mode 'sphere', only isotropic sigma is allowed."

            sphere = self.__make_sphere(radius)
            return convolve(array, sphere, mode="constant", cval=0.0)[roi_slices]

        else:
            raise RuntimeError("Unknown mode %s" % mode)

    def get_context(self):
        if self.mode == "gaussian":
            return tuple((3.0 * s for s in self.sigma))
        elif self.mode == "sphere":
            return self.sigma

    def __outer_product(self, array):
        k = array.shape[0]
        outer = np.einsum("i...,j...->ij...", array, array)
        return outer.reshape((k**2,) + array.shape[1:])

    def __upsample(self, array, f):
        shape = array.shape
        stride = array.strides

        if len(array.shape) == 4:
            sh = (shape[0], shape[1], f, shape[2], f, shape[3], f)
            st = (stride[0], stride[1], 0, stride[2], 0, stride[3], 0)
        else:
            sh = (shape[0], shape[1], f, shape[2], f)
            st = (stride[0], stride[1], 0, stride[2], 0)

        view = as_strided(array, sh, st)

        shapes = [shape[0]]
        [shapes.append(shape[i + 1] * f) for i, j in enumerate(shape[1:])]
        return view.reshape(shapes)
