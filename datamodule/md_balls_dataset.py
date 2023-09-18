from .abstract_md_balls_dataset import BallsDataset, circle
import torch
import os
import numpy as np
import math
from typing import Callable, Optional
import pygame
from pygame import gfxdraw
import colorsys
import utils.general as utils
from .md_balls_dataset_pickle import MDBallsPickleable
log = utils.get_logger(__name__)
from tqdm import tqdm
log_ = False

if "SDL_VIDEODRIVER" not in os.environ:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dsp"

# HSV colours
COLOURS_ = [
    [0.95, 0.6, 0.6],
    # [0.85, 0.6, 0.6],
    [0.75, 0.6, 0.6],
    # [0.65, 0.6, 0.6],
    # [0.55, 0.6, 0.6],
    [0.45, 0.6, 0.6],
    # [0.35, 0.6, 0.6],
    # [0.25, 0.6, 0.6],
    [0.15, 0.6, 0.6],
    # [0.05, 0.6, 0.6],
]

SHAPES_ = [
    "circle",
    # "diamond",
    "square",
    "triangle",
    "heart"
]

PROPERTIES_ = [
    "x",
    "y",
    "c",
    "s",
    "l",
    "p",
]

SCREEN_DIM = 64
Y_SHIFT = 0.0

def draw_shape(
    x_,
    y_,
    surf,
    color=(204, 204, 0),
    radius=0.1,
    screen_width=SCREEN_DIM,
    y_shift=Y_SHIFT,
    offset=None,
    shape="circle",
    rotation_angle=0.
):
    if offset is None:
        offset = screen_width / 2
    scale = screen_width
    x = scale * x_ + offset
    y = scale * y_ + offset

    temp_surf_final = pygame.Surface((screen_width, screen_width), pygame.SRCALPHA)
    if shape != "heart" and shape != "diamond":
        temp_surf_rotation = pygame.Surface((2 * scale * radius, 2 * scale * radius), pygame.SRCALPHA) # for rotations
        temp_surf = pygame.Surface((2 * scale * radius, 2 * scale * radius), pygame.SRCALPHA) # for rotations
    else:
        if shape == "heart":
            temp_surf_rotation = pygame.Surface((2.2 * scale * radius, 2.2 * scale * radius), pygame.SRCALPHA) # for rotations
            temp_surf = pygame.Surface((2.2 * scale * radius, 2.2 * scale * radius), pygame.SRCALPHA) # for rotations
        else:
            temp_surf_rotation = pygame.Surface((3 * scale * radius, 3 * scale * radius), pygame.SRCALPHA) # for rotations
            temp_surf = pygame.Surface((3 * scale * radius, 3 * scale * radius), pygame.SRCALPHA) # for rotations

    if shape == "circle":
        # pygame.draw.circle(surface=temp_surf_rotation, color=color,
        #                center=(0, 0), radius=int(radius * scale))
        # gfxdraw.aacircle(
        #     temp_surf_rotation, int(radius), int(radius), int(radius * scale), color
        #     )
        # gfxdraw.filled_circle(
        #     temp_surf_rotation, int(radius), int(radius), int(radius * scale), color
        # )
        gfxdraw.aacircle(
            surf, int(x), int(y - offset * y_shift), int(radius * scale), color
            )
        gfxdraw.filled_circle(
            surf, int(x), int(y - offset * y_shift), int(radius * scale), color
        )

        # for segmentation mask
        gfxdraw.aacircle(
        temp_surf, int(radius), int(radius), int(radius * scale), color
            )
        gfxdraw.filled_circle(
            temp_surf, int(radius), int(radius), int(radius * scale), color
            )
        # gfxdraw.filled_circle(
        #     temp_surf, int(x), int(y), int(radius * scale), color
        #     )

    elif shape == "square":
        radius = int(radius * scale)*2
        pygame.draw.polygon(surface=temp_surf_rotation, color=color,
                        points=[(int(i), int(j)) for i, j in [(0,0), (radius,0), (radius,radius), (0,radius)]])
        # for segmentation mask
        pygame.draw.polygon(surface=temp_surf, color=color,
                        points=[(int(i), int(j)) for i, j in [(0, 0), (radius,0), (radius,radius), (0,radius)]])
        # pygame.draw.polygon(surface=temp_surf, color=color,
        #                 points=[(int(i), int(j)) for i, j in [(int(x), int(y)), (int(x)+radius,int(y)), (int(x)+radius,int(y)+radius), (int(x),int(y)+radius)]])

    elif shape == "triangle":
        radius = (radius * scale)*2
        pygame.draw.polygon(surface=temp_surf_rotation, color=color,
                        points=[(int(i), int(j)) for i, j in [(radius//2,radius), (radius,0), (0,0)]])
        # for segmentation mask
        pygame.draw.polygon(surface=temp_surf, color=color,
                        points=[(int(i), int(j)) for i, j in [(radius//2,radius), (radius,0), (0, 0)]])
        # pygame.draw.polygon(surface=temp_surf, color=color,
        #                 points=[(int(i), int(j)) for i, j in [(int(x)+radius//2,int(y)+radius), (int(x)+radius,int(y)), (int(x), int(y))]])
    
    elif shape == "diamond":
        radius = (radius * scale) * 1.4
        # pygame.draw.polygon(surface=temp_surf_rotation, color=color,
        #                 # points=[(int(i), int(j)) for i, j in [(radius//2,radius), (radius,0), (radius//2,-radius), (0,0)]])
        #                 points=[(int(i), int(j)) for i, j in [(radius//2, 2 * radius), (radius, radius), (radius//2, 0), (0, radius)]])
        # # for segmentation mask
        # pygame.draw.polygon(surface=temp_surf, color=color,
        #                 # points=[(int(i), int(j)) for i, j in [(radius//2,radius), (radius,0), (radius//2,-radius), (0, 0)]])
        #                 points=[(int(i), int(j)) for i, j in [(radius//2, 2 * radius), (radius, radius), (radius//2, 0), (0, radius)]])

        pygame.draw.polygon(surface=temp_surf_rotation, color=color,
                        # points=[(int(i), int(j)) for i, j in [(radius//2,radius), (radius,0), (radius//2,-radius), (0,0)]])
                        points=[(int(i), int(j)) for i, j in [(radius//2, 2 * radius), (radius, radius), (0, radius)]])
        pygame.draw.polygon(surface=temp_surf_rotation, color=color,
                        # points=[(int(i), int(j)) for i, j in [(radius//2,radius), (radius,0), (radius//2,-radius), (0,0)]])
                        points=[(int(i), int(j)) for i, j in [(radius//2, 0), (radius, radius), (0, radius)]])

        # for segmentation mask
        pygame.draw.polygon(surface=temp_surf, color=color,
                        # points=[(int(i), int(j)) for i, j in [(radius//2,radius), (radius,0), (radius//2,-radius), (0,0)]])
                        points=[(int(i), int(j)) for i, j in [(radius//2, 2 * radius), (radius, radius), (0, radius)]])
        pygame.draw.polygon(surface=temp_surf, color=color,
                        # points=[(int(i), int(j)) for i, j in [(radius//2,radius), (radius,0), (radius//2,-radius), (0,0)]])
                        points=[(int(i), int(j)) for i, j in [(radius//2, 0), (radius, radius), (0, radius)]])

        # pygame.draw.polygon(surface=temp_surf, color=color,
        #                 points=[(int(i), int(j)) for i, j in [(int(x)+radius//2,int(y)+radius), (int(x)+radius,int(y)), (int(x), int(y))]])

    elif shape == "heart":
        radius = (radius * scale)*2
        s = 3.4 # 3.5
        j = 1.33
        offset_x = 3
        pygame.draw.circle(surface=temp_surf_rotation, color=color,
                    center=(offset_x + int(3 * radius /(s * j)), int(radius/(s * j) + radius/2)), radius=int(radius/s))
        pygame.draw.circle(surface=temp_surf_rotation, color=color,
                    center=(offset_x + int(radius/(s*j)), int(radius /(s*j) + radius/2)), radius=int(radius/s))
        pygame.draw.polygon(surface=temp_surf_rotation, color=color,
                        points=[(int(np.floor(i)), int(np.floor(j))) for i, j in [(offset_x + 2*radius/(s*j),0), (offset_x + 2 * radius/(s*j) - radius/2.0,radius/30 + radius/2), (offset_x + 2*radius/(s*j) + radius/2.0,radius/30 + radius/2)]])
        # for segmentation mask
        pygame.draw.circle(surface=temp_surf, color=color,
                    center=(int(3 * radius /(s * j)), int(radius/(s * j) + radius/2)), radius=int(radius/s))
        pygame.draw.circle(surface=temp_surf, color=color,
                    center=(int(radius/(s*j)), int(radius /(s*j) + radius/2)), radius=int(radius/s))
        pygame.draw.polygon(surface=temp_surf, color=color,
                        points=[(int(np.floor(i)), int(np.floor(j))) for i, j in [(2*radius/(s*j),0), (2 * radius/(s*j) - radius/2.0, radius/30 + radius/2), (2*radius/(s*j) + radius/2.0,radius/30 + radius/2)]])
        # pygame.draw.circle(surface=temp_surf, color=color,
        #             center=(int(x)+int(3 * radius /(s * j)), int(y)+int(radius/(s * j) + radius/2)), radius=int(radius/s))
        # pygame.draw.circle(surface=temp_surf, color=color,
        #             center=(int(x)+int(radius/(s*j)), int(y)+int(radius /(s*j) + radius/2)), radius=int(radius/s))
        # pygame.draw.polygon(surface=temp_surf, color=color,
        #                 points=[(int(np.floor(i)), int(np.floor(j))) for i, j in [(int(x)+2*radius/(s*j),int(y)), (int(x) + 2 * radius/(s*j) - radius/2.0,int(y) + radius/30 + radius/2), (int(x) + 2*radius/(s*j) + radius/2.0,int(y) + radius/30 + radius/2)]])

    rotated_surf = pygame.Surface((screen_width, screen_width), pygame.SRCALPHA)
    # Rotate the temporary surface with the rectangle and blit it onto the new surface
    rotated_temp_surf = pygame.transform.rotate(temp_surf_rotation, math.degrees(rotation_angle))
    rotated_temp_surf_rect = rotated_temp_surf.get_rect(center=(int(x), int(y)))
    rotated_surf.blit(rotated_temp_surf, rotated_temp_surf_rect) # now rotations will be around the center and in-place
    surf.blit(rotated_surf, (0, 0))

    temp_surf_pos = (0,0)
    ball_mask = pygame.mask.from_surface(temp_surf)

    # # mask -› surface
    # new_temp_surf = ball_mask.to_surface()
    # # do the same flip as the one occurring for the screen
    # new_temp_surf = pygame.transform.flip(new_temp_surf, False, True)
    # new_temp_surf.set_colorkey((0,0,0))
    
    # -------
    rotated_surf = pygame.Surface((screen_width, screen_width), pygame.SRCALPHA)
    # Rotate the temporary surface with the rectangle and blit it onto the new surface
    rotated_temp_surf = pygame.transform.rotate(temp_surf, math.degrees(rotation_angle))
    rotated_temp_surf.set_colorkey((0,0,0))
    rotated_temp_surf_rect = rotated_temp_surf.get_rect(center=(int(x), int(y)))
    rotated_surf.blit(rotated_temp_surf, rotated_temp_surf_rect) # now rotations will be around the center and in-place
    rotated_surf = pygame.transform.flip(rotated_surf, False, False)
    temp_surf_final.blit(rotated_surf, (0, 0))
    # -------

    return np.transpose(np.array(pygame.surfarray.pixels3d(temp_surf_final)), axes=(1, 0, 2))[:, :, :1] # [screen_width, screen_width, 1]


    
class MDBalls(BallsDataset):
    """
    """

    screen_dim = SCREEN_DIM

    def __init__(
        self,
        transform: Optional[Callable] = None, # type: ignore
        augmentations: list = [],
        human_mode: bool = True,
        n_balls_invariant: int = 1,
        n_balls_spurious: int = 1,
        num_samples: int = 20000,
        num_domains: int = 2,
        invariant_low: list = [0.0, 0.0],
        invariant_high: list = [1.0, 1.0],
        **kwargs,
    ):
        super(MDBalls, self).__init__(transform
                                                , augmentations
                                                , human_mode
                                                , num_samples
                                                ,**kwargs
                                                )
        if transform is None:
            def transform(x):
                return x

        self.transform = transform

        self.n_balls_invariant = n_balls_invariant
        self.n_balls_spurious = n_balls_spurious
        self.n_balls = n_balls_invariant + n_balls_spurious
        self.num_domains = num_domains
        self.domain_lengths = [1 / num_domains] * num_domains
        self.ball_z_dim = len(PROPERTIES_)
        self.ball_size = kwargs.get("ball_size", 0.1)
        self.invariant_low = invariant_low
        self.invariant_high = invariant_high

        self.same_color = kwargs.get("same_color")
        self.properties_list = kwargs.get("properties_list") # a subset of ["x","y","c","s"] preserving the order
        self.target_property_indices = [i for i,p in enumerate(PROPERTIES_) if p in self.properties_list]
        self.non_target_property_indices = [i for i,p in enumerate(PROPERTIES_) if p not in self.properties_list]
        self.pickleable_dataset_params = {}
        self.data = self._generate_data()
        print(f"self.pickleable_dataset_params: {self.pickleable_dataset_params}")
        self.pickleable_dataset = MDBallsPickleable(self.data, self.transform, **self.pickleable_dataset_params)


    def draw_sample(self, z_all):
        self._setup(SCREEN_DIM)
        
        if log_:
            print(f"===========\n===========\n===========\nz_all:\n{z_all}\n===========\n===========\n===========")
        
        hsv_colours = [COLOURS_[z_all[i,2].astype(int)] for i in range(z_all.shape[0])]
        shapes = [z_all[i,3].astype(int) for i in range(z_all.shape[0])]
        sizes = [z_all[i,4] for i in range(z_all.shape[0])]

        # filling z_all with colour hues at dimension 2
        z_all[:, 2] = np.array(hsv_colours)[:, 0]
        # note the multiplication by 255., because draw_scene works with rgb colours in the range [0, 255.]
        rgb_colours = [[255.*channel for channel in colorsys.hls_to_rgb(*c)] for c in hsv_colours]

        # filling z_all with shape indices at dimension 3
        z_all[:, 3] = np.array(shapes)
        
        # filling z_all with size values at dimension 4
        z_all[:, 4] = np.array(sizes)

        # segmentation_mask: [n_balls+1, screen_width, screen_width, 1]; segmentation_mask[0] is the background mask
        x, segmentation_masks = self.draw_scene(z_all, rgb_colours)
        
        # dividing z_all[:, 3] (shape dimension) by the number of shapes so the latent
        # becomes nicer and close to the rest of the features.
        z_all[:, 3] /= len(SHAPES_)

        x = self.transform(x)
        z = z_all[..., self.target_property_indices].copy()

        self._teardown()

        return {"z": z.flatten()
                , "z_invariant": z_all[:self.n_balls_invariant, self.target_property_indices].copy().flatten()
                , "z_spurious": z_all[self.n_balls_invariant:, self.target_property_indices].copy().flatten()
                , "image": x
                , "segmentation_mask": segmentation_masks
                , "coordinate": z_all[:, :2].flatten()
                , "color": z_all[:, 2].flatten()
                }

    def _generate_data(self):
        data = []
        
        z_all = np.zeros((self.num_samples, self.n_balls, len(PROPERTIES_)))
        # build domain grids
        # domain_lows = [num_domains, n_balls_spurious, ball_z_dim]
        # domain_highs = [num_domains, n_balls_spurious, ball_z_dim]
        self.domain_lows, self.domain_highs = self._build_domain_grids()

        domain_mask = torch.zeros(self.num_samples, 1)
        start = 0
        for domain_idx in range(self.num_domains):
            domain_size = int(self.domain_lengths[domain_idx] * self.num_samples)
            end = domain_size + start
            domain_mask[start:end] = domain_idx
            start = end
        # sample z_spurious for each domain
        z_spurious = self._sample_z_spurious(self.domain_lows, self.domain_highs, domain_mask)
        z_all[:, self.n_balls_invariant:, :] = z_spurious

        # sample z_invariant for all domains
        z_invariant = self._sample_z_invariant(z_all)
        z_all[:, :self.n_balls_invariant, :] = z_invariant

        # draw scenes with z_all for all samples
        for i in tqdm(range(self.num_samples)):
            sample = self.draw_sample(z_all[i])
            sample["domain"] = domain_mask[i]
            data.append(sample)
        
        # normalize the image key values by its min and max. min and max are 
        # 3 dimensional (rgb)
        # concatenate all images
        images = np.concatenate([d["image"] for d in data], axis=0) # [num_samples, screen_width, screen_width, 3]
        # find the min of the last dimension (rgb) for all images
        self.min_ = images.min(axis=(0,1,2)) # [3]
        # find the max of the last dimension (rgb) for all images
        self.max_ = images.max(axis=(0,1,2)) # [3]
        # find the mean of the last dimension (rgb) for all images
        self.mean_ = images.mean(axis=(0,1,2)) # [3]
        # find the std of the last dimension (rgb) for all images
        self.std_ = images.std(axis=(0,1,2)) # [3]
        # # normalize all images by min and max
        # for i, d in enumerate(data):
        #     data[i]["image"] = (d["image"] - self.min_) / (self.max_ - self.min_)
        self.pickleable_dataset_params["mean"] = self.mean_
        self.pickleable_dataset_params["std"] = self.std_
        # normalize all images by mean and std
        for i, d in enumerate(data):
            data[i]["image"] = (d["image"] - self.mean_) / self.std_

        return data

    def __getitem__(self, idx):
        return {"image": self.data[idx]["image"], "z": self.data[idx]["z"], "z_invariant": self.data[idx]["z_invariant"], "z_spurious": self.data[idx]["z_spurious"], "domain": self.data[idx]["domain"], "color": self.data[idx]["color"]}

    def _build_domain_grids(self):
        domain_lows = np.zeros((self.num_domains, self.n_balls_spurious, self.ball_z_dim))
        domain_highs = np.zeros((self.num_domains, self.n_balls_spurious, self.ball_z_dim))

        # first two dimensions correspond to x,y coordinates
        for i in range(self.num_domains):

            # for each domain, we sample the low and high of x,y coordinates of the balls
            domain_low_ = np.random.uniform(self.ball_size, 1.0 - self.ball_size, size=(self.n_balls_spurious, 2))
            domain_high_ = np.random.uniform(self.ball_size, 1.0 - self.ball_size, size=(self.n_balls_spurious, 2))
            # resample lows and highs until high_ for each dimension is greater than low_ for the same dimension
            while (domain_high_ < domain_low_).any() or (domain_high_ < domain_low_ + 2 * self.ball_size).any():
                domain_low_ = np.random.uniform(self.ball_size, 1.0 - self.ball_size, size=(self.n_balls_spurious, 2))
                domain_high_ = np.random.uniform(self.ball_size, 1.0 - self.ball_size, size=(self.n_balls_spurious, 2))

            # make sure that the intersection of any pair of grids is at least twice the size of the balls
            # this is to make sure that the balls are not initialized very close to each other
            # intersection sides of each pair of grids
            intersection_sides = np.zeros((self.n_balls_spurious, self.n_balls_spurious, 2)) + np.inf
            for j in range(self.n_balls_spurious):
                for k in range(j+1, self.n_balls_spurious):
                    # check if there is any intersection between the two grids
                    if (domain_low_[j] < domain_high_[k]).all() and (domain_high_[j] > domain_low_[k]).all():
                        intersection_sides[j, k] = np.minimum(domain_high_[j], domain_high_[k]) - np.maximum(domain_low_[j], domain_low_[k])
                    else:
                        intersection_sides[j, k] = np.inf
            # check if the intersection is at least twice the size of the balls, and if not resample
            while_loop_threshold = 100
            while (intersection_sides < 4 * self.ball_size).any() and while_loop_threshold > 0:
                while_loop_threshold -= 1
                domain_low_ = np.random.uniform(self.ball_size, 1.0 - self.ball_size, size=(self.n_balls_spurious, 2))
                domain_high_ = np.random.uniform(self.ball_size, 1.0 - self.ball_size, size=(self.n_balls_spurious, 2))
                while (domain_high_ < domain_low_).any() or (domain_high_ < domain_low_ + 2 * self.ball_size).any():
                    domain_low_ = np.random.uniform(self.ball_size, 1.0 - self.ball_size, size=(self.n_balls_spurious, 2))
                    domain_high_ = np.random.uniform(self.ball_size, 1.0 - self.ball_size, size=(self.n_balls_spurious, 2))
                for j in range(self.n_balls_spurious):
                    for k in range(j+1, self.n_balls_spurious):
                        if (domain_low_[j] < domain_high_[k]).all() and (domain_high_[j] > domain_low_[k]).all():
                            intersection_sides[j, k] = np.minimum(domain_high_[j], domain_high_[k]) - np.maximum(domain_low_[j], domain_low_[k])
                        else:
                            intersection_sides[j, k] = np.inf

            if while_loop_threshold == 0:
                if log_:
                    print(f"_build_domain_grids reached {while_loop_threshold} attempts.")
                    log.info(f"_build_domain_grids reached {while_loop_threshold} attempts.")

            domain_lows[i, :, :2] = domain_low_
            domain_highs[i, :, :2] = domain_high_

        return domain_lows, domain_highs
    
    def _sample_z_spurious(self, domain_lows, domain_highs, domain_mask):
        z_spurious = np.zeros((self.num_samples, self.n_balls_spurious, self.ball_z_dim))
        for domain_idx in range(self.num_domains):
            domain_mask_ = (domain_mask == domain_idx).squeeze()            
            domain_length = len(domain_mask_[domain_mask_ == True])
            for j in range(self.n_balls_spurious):
                # note that domain_lows and domain_highs should correspond to the boundary of where objects fall
                # so we should adjust for half the size of ball in initialization so that all objects
                # fall in the domain boundaries
                z_spurious[domain_mask_, j, 0] = np.random.uniform(domain_lows[domain_idx, j, 0] + self.ball_size, domain_highs[domain_idx, j, 0] - self.ball_size, domain_length)
                z_spurious[domain_mask_, j, 1] = np.random.uniform(domain_lows[domain_idx, j, 1] + self.ball_size, domain_highs[domain_idx, j, 1] - self.ball_size, domain_length)
                z_spurious[domain_mask_, j, 2] = self.n_balls_invariant + j # np.random.choice(range(len(COLOURS_)), domain_length)
                z_spurious[domain_mask_, j, 3] = 0 # np.random.choice(range(len(SHAPES_)), domain_length)
                z_spurious[domain_mask_, j, 4] = self.ball_size # np.random.uniform(self.min_size, self.max_size, domain_length)
                z_spurious[domain_mask_, j, 5] = 0. # np.random.uniform(self.min_phi, self.max_phi, domain_length)
            # resample the first two dimensions of samples where the balls are initialized very close to each other
            z_spurious_replace = z_spurious[domain_mask_].copy()
            for i, sample in enumerate(z_spurious[domain_mask_]):
                # sample: [n_balls_spurious, ball_z_dim]
                duplicate_coordinates_threshold = self.ball_size * 2
                sampled_coordinates_distance_matrix = np.linalg.norm(sample[:, :2][None, :] - sample[:, :2][:, None], axis=-1)
                duplicate_mask = np.triu(sampled_coordinates_distance_matrix<duplicate_coordinates_threshold).sum(-1)>1
                while_loop_threshold = 100
                while duplicate_mask.any():
                    while_loop_threshold -= 1
                    sample[duplicate_mask, :2] = np.random.uniform(domain_lows[domain_idx, duplicate_mask, :2] + self.ball_size, domain_highs[domain_idx, duplicate_mask, :2] - self.ball_size, (duplicate_mask.sum(), 2))
                    sampled_coordinates_distance_matrix = np.linalg.norm(sample[:, :2][None, :] - sample[:, :2][:, None], axis=-1)
                    duplicate_mask = np.triu(sampled_coordinates_distance_matrix<duplicate_coordinates_threshold).sum(-1)>1
                if while_loop_threshold == 0:
                    if log_:
                        print(f"_sample_z_spurious reached {while_loop_threshold} attempts.")
                        log.info(f"_sample_z_spurious reached {while_loop_threshold} attempts.")
                # fill z_spurious with the new sample
                z_spurious_replace[i] = sample
            z_spurious[domain_mask_] = z_spurious_replace

        return z_spurious

    def _sample_z_invariant(self, z_all):
        z_invariant = np.zeros((self.num_samples, self.n_balls_invariant, self.ball_z_dim))
        for i in range(self.num_samples):
            # note that domain_lows and domain_highs should correspond to the boundary of where objects fall
            # so we should adjust for half the size of ball in initialization so that all objects
            # fall in the image boundaries
            z_invariant[i, :, 0] = np.random.uniform(self.invariant_low[0] + self.ball_size, self.invariant_high[0] - self.ball_size, self.n_balls_invariant)
            z_invariant[i, :, 1] = np.random.uniform(self.invariant_low[1] + self.ball_size, self.invariant_high[1] - self.ball_size, self.n_balls_invariant)
            z_invariant[i, :, 2] = list(range(self.n_balls_invariant)) # np.random.choice(range(len(COLOURS_)), self.n_balls_invariant)
            z_invariant[i, :, 3] = 0 # np.random.choice(range(len(SHAPES_)), self.n_balls_invariant)
            z_invariant[i, :, 4] = self.ball_size # np.random.uniform(self.min_size, self.max_size, self.n_balls_invariant)
            z_invariant[i, :, 5] = 0. # np.random.uniform(self.min_phi, self.max_phi, self.n_balls_invariant)
            # resample the first two dimension of samples where the balls are initialized very close to each other
            # make sure to compare the invariant balls together and with the spurious balls from z_all
            duplicate_coordinates_threshold = self.ball_size * 2
            sample_ball_coordinates = np.concatenate([z_invariant[i, :, :2], z_all[i, :, :2]], axis=0) # [n_balls_spurious + n_balls_invariant, 2]
            sampled_coordinates_distance_matrix = np.linalg.norm(sample_ball_coordinates[None, :] - sample_ball_coordinates[:, None], axis=-1) # [n_balls_spurious + n_balls_invariant, n_balls_spurious + n_balls_invariant]
            duplicate_mask = np.triu(sampled_coordinates_distance_matrix<duplicate_coordinates_threshold)[:self.n_balls_invariant].sum(-1)>1
            while_loop_threshold = 100
            while duplicate_mask.any():
                while_loop_threshold -= 1
                z_invariant[i, duplicate_mask, :2] = np.random.uniform(np.array(self.invariant_low) + self.ball_size, np.array(self.invariant_high) - self.ball_size, (duplicate_mask.sum(), 2)).squeeze()
                sample_ball_coordinates = np.concatenate([z_invariant[i, :, :2], z_all[i, :, :2]], axis=0)
                sampled_coordinates_distance_matrix = np.linalg.norm(sample_ball_coordinates[None, :] - sample_ball_coordinates[:, None], axis=-1)
                duplicate_mask = np.triu(sampled_coordinates_distance_matrix<duplicate_coordinates_threshold)[:self.n_balls_invariant].sum(-1)>1
            if while_loop_threshold == 0:
                if log_:
                    print(f"_sample_z_invariant reached {while_loop_threshold} attempts.")
                    log.info(f"_sample_z_invariant reached {while_loop_threshold} attempts.")
        return z_invariant

    def draw_scene(self, z, colours=None):
        self.surf.fill((255, 255, 255))
        # getting the background segmentation mask
        self.bg_surf = pygame.Surface((self.screen_dim, self.screen_dim), pygame.SRCALPHA)

        obj_masks = []
        if z.ndim == 1:
            z = z.reshape((1, 2))
        if colours is None:
            colours = [COLOURS_[3]] * z.shape[0]
        for i in range(z.shape[0]):
            obj_masks.append(
                draw_shape(
                    z[i, 0],
                    z[i, 1],
                    self.surf,
                    color=colours[i],
                    radius=z[i,4],
                    screen_width=self.screen_dim,
                    y_shift=0.0,
                    offset=0.0,
                    shape=SHAPES_[int(z[i,3])],
                    rotation_angle=z[i,5]
                )
            )
            _ = draw_shape(
                z[i, 0],
                z[i, 1],
                self.bg_surf,
                color=colours[i],
                radius=z[i,4],
                screen_width=self.screen_dim,
                y_shift=0.0,
                offset=0.0,
                shape=SHAPES_[int(z[i,3])],
                rotation_angle=z[i,5]
            )

        bg_surf_pos = (0,0)
        bg_mask = pygame.mask.from_surface(self.bg_surf)
        bg_mask.invert() # so that mask bits for balls are cleared and the bg gets set.

        # mask -› surface
        new_bg_surf = bg_mask.to_surface()
        new_bg_surf.set_colorkey((0,0,0))
        # do the same flip as the one occurring for the screen
        new_bg_surf = pygame.transform.flip(new_bg_surf, False, True)

        # print(np.array(pygame.surfarray.pixels3d(new_bg_surf)).shape)
        # bg_mask = np.array(pygame.surfarray.pixels3d(new_bg_surf))[:, :, :1] # [screen_width, screen_width, 1]
        bg_mask = np.transpose(np.array(pygame.surfarray.pixels3d(new_bg_surf)), axes=(1, 0, 2))[:, :, :1] # [screen_width, screen_width, 1]
        # ------------------------------------------ #
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.human_mode:
            pygame.display.flip()
        return (
            np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
                )
            , np.array([bg_mask] + obj_masks)
        )


        # if "c" in self.properties_list:
        #         z_all_1[ball_idx, 2] = np.random.choice(range(len(COLOURS_)), 1)
        #     else:
        #         z_all_1[ball_idx, 2] = 0
        #     if "s" in self.properties_list:
        #         z_all_1[ball_idx, 3] = np.random.choice(range(len(SHAPES_)), 1)
        #     else:
        #         z_all_1[ball_idx, 3] = 0
        #     if "l" in self.properties_list:
        #         z_all_1[ball_idx, 4] = np.random.uniform(self.min_size, self.max_size, 1)
        #     else:
        #         z_all_1[ball_idx, 4] = (self.min_size + self.max_size)/2.
        #     if "p" in self.properties_list:
        #         z_all_1[ball_idx, 5] = np.random.uniform(self.min_phi, self.max_phi, 1)
        #     else:
        #         z_all_1[ball_idx, 5] = 0
        
        # # sample colours for the rest of the balls
        # # if colour is among the target properties then it should be picked at random, o.w. it
        # # should be fixed
        # if "c" in self.properties_list: # colour is among the targets
        #     replace = True if n_balls-1 > num_colours else False
        #     colour_indices = np.random.choice(range(num_colours), n_balls-1, replace=False).astype(int)
        # else:
        #     if not self.same_color:
        #         colour_indices = np.arange(1, n_balls) # n_balls-1 fixed colours
        #     else:
        #         colour_indices = np.zeros(n_balls-1) # n_balls-1 same colour since we want to remove its effect
        # z_all[idx_mask, 2] = colour_indices
        # # z_all[idx_mask, 2] = hsv_colours[idx_mask, 0]
        
        # # sample shapes for the rest of the balls
        # # if shape is among the target properties then it should be picked at random, o.w. it
        # # should be fixed
        # if "s" in self.properties_list: # shape is among the targets
        #     shape_indices = np.random.choice(range(num_shapes), n_balls-1)
        # else:
        #     if not self.same_shape:
        #         shape_indices = np.arange(1, n_balls) # n_balls-1 fixed shapes
        #     else:
        #         shape_indices = np.zeros((n_balls-1,)) # n_balls-1 same shapes since we want to remove its effect
        #     # shape_indices = np.zeros((n_balls-1,))
        # z_all[idx_mask, 3] = shape_indices

        # # sample sizes for the rest of the balls
        # # if size is among the target properties then it should be picked at random, o.w. it
        # # should be fixed
        # if "l" in self.properties_list: # shape is among the targets
        #     sizes = np.random.uniform(self.min_size, self.max_size, n_balls-1)
        # else:
        #     sizes = np.zeros((n_balls-1,)) + (self.min_size+self.max_size)/2.
        # z_all[idx_mask, 4] = sizes
        
        # # sample rotation angles for the rest of the balls
        # # if rotation angle is among the target properties then it should be picked at random, o.w. it
        # # should be fixed
        # if "p" in self.properties_list: # rotation angle is among the targets
        #     rotation_angles = np.random.uniform(self.min_phi, self.max_phi, n_balls-1)
        # else:
        #     rotation_angles = np.zeros((n_balls-1,))
        # z_all[idx_mask, 5] = rotation_angles

    def renormalize(self):
        # for t in self.transform.transforms:
        #     if t.__class__.__name__ == "Standardize":
        #         """Renormalize from [-1, 1] to [0, 1]."""
        #         return lambda x: x / 2.0 + 0.5
        #     else:
        #         return lambda x: x
        return lambda x: x * self.std_ + self.mean_
        
