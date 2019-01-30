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
    # lots of modulo, TODO check if removal is necessary
    angle = 360 / n
    for i in range(n-1):
        yield angle*i, angle*(i+1)
    else:
        # ensure that final sector ends at 360 degrees,
        # in case of rounding error
        yield angle*(n-1), 360


def create_frame(img_matrices, offset):
    n = len(img_matrices)
    full_matrix = None

    angle_ranges = [(a+offset, b+offset)  for a,b in gen_start_angle_ranges(n)]
    
    for i, (matrix, angle_range) in enumerate(zip(img_matrices, angle_ranges)):
        
        # individual masked image sector
        width, height = matrix.shape[:2]
        # radius is width // 2. This is an assumption that height ~= width, radius ~= width
        mask = sector_mask(matrix.shape, (width//2, height//2), width//2, angle_range)
        matrix[~mask] = 0
        if full_matrix is None:
            full_matrix = matrix
        else:
            # not checked for efficiency. is this index based?
            full_matrix = np.where(matrix == 0, full_matrix, matrix)

    # TEMPORARY INVERSION until i debug...
    return cv2.bitwise_not(full_matrix) # uninvert image... WHERE DID IT BECOME INVERTED????



def main():
    filepaths = glob("data/test/*.png")

    FPS = 10
    ROTATION_PER_FRAME = 0.5
    VIDEO_SHAPE = (640,640)
    
    # init video for mp4, using correct fourcc code
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter('data/test/video.mp4', fourcc, FPS, VIDEO_SHAPE, 1)

    # iterate through different offsets, to rotate each sector over the video
    for offset in np.arange(0, 360+ROTATION_PER_FRAME, ROTATION_PER_FRAME):
        # create the frame full of sectors. it loads the images every time to avoid weird graphical glitches.
        frame = cv2.resize(create_frame([cv2.imread(fp) for fp in filepaths], offset), VIDEO_SHAPE)
        
        # cv2.imwrite(f"data/test/res/{offset}.png", frame)
        video.write(frame)
        print(f"Frame created for offset {offset} degrees.")
        
    video.release()


if __name__ == '__main__':
    main()