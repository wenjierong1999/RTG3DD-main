import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


img_path = "/data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/vis_tools/failure_case_output/b9ee1532-64f0-4c90-a88f-85ed835e7cb7/b9ee1532-64f0-4c90-a88f-85ed835e7cb7_view_1.png"
img = plt.imread(img_path)


x1, x2, y1, y2 = 500, 700, 350, 550
margin = 5

fig, axes = plt.subplots(1, 2, figsize=(12, 6))


axes[0].imshow(img)
rect = patches.Rectangle((x1+margin, y1+margin), (x2 - x1) - 2*margin, (y2 - y1) - 2*margin,
                         linewidth=3, edgecolor='red', facecolor='none')
axes[0].add_patch(rect)
axes[0].axis('off')


zoomed_img = img[y1:y2, x1:x2]
h, w = zoomed_img.shape[:2]
axes[1].imshow(zoomed_img, extent=[0, w, 0, h])  # 坐标系与像素对应
rect_zoom = patches.Rectangle((0, 0), w, h,
                              linewidth=3, edgecolor='red', facecolor='none')
axes[1].add_patch(rect_zoom)
axes[1].set_xlim(0, w)
axes[1].set_ylim(0, h)
axes[1].axis('off')


fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)


output_path = "zoomed_image_fixed.png"
plt.savefig(output_path, dpi=500, pad_inches=0.2)
plt.close()