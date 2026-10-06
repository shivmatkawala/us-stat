import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Charts to include (in a logical layout)
chart_files = [
    "output/revenue_by_category.png",
    "output/monthly_revenue.png",
    "output/top_products_revenue.png",
    "output/payment_method_revenue.png",
    "output/order_status_counts.png",
    "output/top_cities_revenue.png",
    "output/gender_revenue_donut.png",
]

# Validate existence and load images
loaded_images = []
missing = []
for f in chart_files:
    if os.path.exists(f):
        try:
            img = mpimg.imread(f)
            loaded_images.append((f, img))
        except Exception as e:
            missing.append((f, str(e)))
    else:
        missing.append((f, "File not found"))

print("=== Charts to compose ===")
print({"requested": chart_files, "loaded_count": len(loaded_images), "missing": missing})

# If none loaded, exit gracefully
if not loaded_images:
    print("No charts available to compose. Aborting dashboard creation.")
else:
    # Create dashboard figure
    # Layout: Title row + 2 rows x 3 columns grid; last row uses 1 column for donut centered with padding
    fig = plt.figure(figsize=(18, 14))  # wide canvas
    fig.suptitle("Ecommerce Orders Dashboard", fontsize=20, fontweight="bold", y=0.98)

    # Helper to add an image to a specific grid cell
    def add_image_subplot(grid_spec, position, img, title=None):
        ax = fig.add_subplot(grid_spec[position])
        ax.imshow(img)
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=12, pad=6)

    # Build GridSpec with 3 rows (title handled via suptitle), but we allocate 3 content rows
    gs = fig.add_gridspec(nrows=3, ncols=3, height_ratios=[1, 1, 1], hspace=0.25, wspace=0.08)

    # Map images into slots
    # Row 1
    slots = []
    for r in range(3):
        for c in range(3):
            slots.append((r, c))
    slot_idx = 0

    # Place first six images in 2 rows x 3 columns
    titles_map = {
        "output/revenue_by_category.png": "Revenue by Category",
        "output/monthly_revenue.png": "Monthly Revenue",
        "output/top_products_revenue.png": "Top Products by Revenue",
        "output/payment_method_revenue.png": "Revenue by Payment Method",
        "output/order_status_counts.png": "Order Status Counts",
        "output/top_cities_revenue.png": "Top Cities by Revenue",
        "output/gender_revenue_donut.png": "Gender-wise Revenue",
    }

    # First six images go to first two rows
    for i in range(min(6, len(loaded_images))):
        f, img = loaded_images[i]
        r, c = slots[slot_idx]
        add_image_subplot(gs, (r, c), img, titles_map.get(f))
        slot_idx += 1

    # Last row: center the donut chart; if not available, place next available image
    # Find donut image first
    donut_idx = None
    for i, (f, img) in enumerate(loaded_images):
        if f.endswith("gender_revenue_donut.png"):
            donut_idx = i
            break

    if donut_idx is not None:
        f, img = loaded_images[donut_idx]
    else:
        # fallback: next available image beyond the first six
        if len(loaded_images) > 6:
            f, img = loaded_images[6]
        else:
            # reuse revenue_by_category if nothing else left
            f, img = loaded_images[0]

    # Place the donut centered (span all columns in row 3)
    ax_bottom = fig.add_subplot(gs[2, :])
    ax_bottom.imshow(img)
    ax_bottom.axis("off")
    ax_bottom.set_title(titles_map.get(f, "Chart"), fontsize=12, pad=6)

    # Save the composed dashboard
    out_path = "output/dashboard_ecommerce_orders.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("=== Dashboard saved ===")
    print({"path": out_path, "figsize": (18, 14), "charts_used": [f for f, _ in loaded_images]})