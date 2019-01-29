import numpy as np

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
    angle = 360 / n
    for i in range(n-1):
        yield angle*i, angle*(i+1)
    else:
        # ensure that final sector ends at 360 degrees,
        # in case of rounding error
        yield angle*(n-1), 360  



from matplotlib import pyplot as plt
import seaborn as sns
import cv2
from glob import glob

filepaths = glob("data/test/*.png")

n = len(filepaths)
fig, ax = plt.subplots()

main_matrix = None
for fp, angle_range in zip(filepaths, gen_start_angle_ranges(n)):
    # individual masked image sector
    matrix = cv2.imread(fp)
    width, height = matrix.shape[:2]
    # radius is width // 2. This is an assumption that height ~= width, radius ~= width
    mask = sector_mask(matrix.shape, (width//2, height//2), width//2, angle_range)
    matrix[~mask] = 0

    if main_matrix is None:
        main_matrix = matrix
    else:
        # not checked for efficiency. is this index based?
        main_matrix = np.where(main_matrix != 0, main_matrix, matrix)
    
# plot
ax.imshow(main_matrix) # test
sns.despine(ax=ax, left=True, bottom=True)
ax.set_yticks([])
ax.set_xticks([])
plt.show()