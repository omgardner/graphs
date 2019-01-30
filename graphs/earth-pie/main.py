import numpy as np
from matplotlib import pyplot as plt
import cv2
from glob import glob

def sector_mask(shape,centre,radius,angle_range):

    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    https://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)
    # ensure stop angle > start angle
    if tmax < tmin:
        tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin
    
    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask


def gen_start_angle_ranges(n):
    # asign equal angle range to each file via generator
    # `offset` rotates all angle ranges by `offset` degrees
    # module not necessary as it is handled in the sector masking already (2π == 4π == 6π == ...)
    angle = 360 / n
    for i in range(n-1):
        yield angle*i, angle*(i+1)
    else:
        # ensure that final sector ends at 360 degrees,
        # in case of rounding error
        yield angle*(n-1), 360


def create_frame(img_matrices, offset, mask_kwargs):
    """ return the complete frame with sector sections
    img_matrices: 
        numpy matrices representing images to sectorise
    offset:
        rotation offset of images by `offset` degrees
    mask_kwargs:
        kwargs dict for sector_mask function
    """
    n = len(img_matrices)
    full_matrix = None

    angle_ranges = [(a+offset, b+offset)  for a,b in gen_start_angle_ranges(n)]
    
    for matrix, angle_range in zip(img_matrices, angle_ranges):
        
        # individual masked image sector
        width, height = matrix.shape[:2]
        # radius is width // 2. This is an assumption that height ~= width, radius ~= width
        mask = sector_mask(angle_range=angle_range, **mask_kwargs)
        matrix[~mask] = 0
        # add to complete image matrix
        if full_matrix is None:
            full_matrix = matrix
        else:
            # not checked for efficiency. is this index based?
            full_matrix = np.where(full_matrix != 0, full_matrix, matrix)

    # TEMPORARY INVERSION until i debug...
    return full_matrix # cv2.bitwise_not(full_matrix) # uninvert image... sometimes



def main():

    DIRPATH = "data/test"
    # get all images with png or jpg file extension
    filepaths = glob(f"{DIRPATH}/*.png")
    print(filepaths)

    FPS = 24
    ROTATION_PER_FRAME = 5
    VIDEO_SHAPE = (640,640)
    W,H = VIDEO_SHAPE

    mask_kwargs = {
        "shape": VIDEO_SHAPE,
        "centre": (W//2, H//2),
        "radius": int(3*max(VIDEO_SHAPE)/4) # 3/4 of height
    }

    # init video for mp4, using correct fourcc code
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(f'{DIRPATH}/video.mp4', fourcc, FPS, VIDEO_SHAPE, 1)

    # init images
    images = [cv2.resize(cv2.imread(fp), VIDEO_SHAPE) for fp in filepaths]
    print(f"{len(images)} images loaded, resized to ({W}px, {H}px).")

    # iterate through different offsets, to rotate each sector over the video
    for offset in np.arange(0, 360+ROTATION_PER_FRAME, ROTATION_PER_FRAME):
        # create the frame full of sectors. it copies the images every time to avoid weird graphical glitches.
        frame = create_frame([img.copy() for img in images], offset, mask_kwargs)
        #cv2.imshow(frame, 0)
        # cv2.imwrite(f"data/test/res/{offset}.png", frame)
        video.write(frame)
        print(f"Frame created for offset {offset} degrees.")
        
    video.release()


if __name__ == '__main__':
    main()