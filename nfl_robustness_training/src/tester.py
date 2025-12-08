
from integrated_reachable_sim import setup_analyzer
from integrated_reachable_sim import ReachabilityTester
from integrated_reachable_sim import CalculationType
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import numpy as np

def test():
    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')

    print("\nCreating interactive tester...")
    tester = ReachabilityTester(analyzer)

    print("\n" + "=" * 20 + " Initial State " + "=" * 20)
    print(tester.horizons[0])
    t = 0
    while t<15:
        tester.concrete(t,t+3)
        tester.real_state_empirical(t,t+1)
        tester.horizons[t].list_calculations()

        t+=1

def test1():
    analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')

    print("\nCreating interactive tester...")
    tester = ReachabilityTester(analyzer)
    
    print("\n" + "=" * 20 + " Initial State " + "=" * 20)
    print(tester.horizons[0])
    t = 0
    while t<10:

        tester.concrete(t,t+1)
        # tester.concrete(t,t+3)
        # tester.concrete(t,t+4)
        # tester.concrete(t,t+5)
        # tester.concrete(t,t+6)
        # tester.concrete(t,t+7)
        # tester.concrete(t,t+8)
        # tester.concrete(t,t+9)
        tester.real_state_empirical(t,t+1)
        # tester.horizons[t].list_calculations()

        t+=1

def animate():
    """Create animation of reachability propagation following test pattern"""
    print("=" * 80)
    print("CREATING ANIMATION")
    print("=" * 80)

    analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')

    # Create tester without dynamic plot (we'll save frames instead)
    tester = ReachabilityTester(analyzer)

    print("\nGenerating animation frames...")

    frames = []
    colors_map = {
        CalculationType.CONCRETE: 'blue',
        CalculationType.SYMBOLIC: 'green',
        CalculationType.EMPIRICAL: 'purple'
    }

    # Generate frames following the same pattern as test
    t = 0
    max_t = 12  # Stop at 12 to avoid going past 15 with t+3

    while t <= max_t:
        print(f"Generating frame for t={t}")

        # Compute reachable sets - ONLY CONCRETE
        if t + 3 <= 15:
            tester.concrete(t, t+1)
            pass

        tester.real_state_empirical(t, t+1) #########################

        # Create NEW figure for each frame
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)

        ax.set_title(f'Reachability Analysis - Time t={t}', fontsize=14, fontweight='bold')
        ax.set_xlabel('State 1', fontsize=12)
        ax.set_ylabel('State 2', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot all horizons
        all_bounds = []

        # For current timestep, show individual calculation bounds
        if t + 3 in tester.horizons:
            horizon_t3 = tester.horizons[t + 3]

            # Draw individual concrete propagation steps (t+1, t+2, t+3)
            for calc_id, calc_info in horizon_t3.calculations.items():
                if calc_info['calc_type'] == CalculationType.CONCRETE:
                    bounds = calc_info['bounds']
                    all_bounds.append(bounds)
                    step_num = calc_info['step_size']

                    # Color based on step
                    if step_num == 1:
                        color = 'lightblue'
                        label = f'Concrete t+1'
                        alpha = 0.3
                    elif step_num == 2:
                        color = 'cornflowerblue'
                        label = f'Concrete t+2'
                        alpha = 0.4
                    elif step_num == 3:
                        color = 'blue'
                        label = f'Concrete t+3'
                        alpha = 0.5
                    else:
                        continue

                    rect = Rectangle(
                        bounds[:2, 0],
                        bounds[0, 1] - bounds[0, 0],
                        bounds[1, 1] - bounds[1, 0],
                        edgecolor=color,
                        facecolor='none',
                        alpha=alpha,
                        linewidth=2,
                        linestyle='--',
                        label=label
                    )
                    ax.add_patch(rect)

        # Draw empirical bound at t+1
        if t + 1 in tester.horizons:
            horizon_t1 = tester.horizons[t + 1]
            for calc_id, calc_info in horizon_t1.calculations.items():
                if calc_info['calc_type'] == CalculationType.EMPIRICAL:
                    bounds = calc_info['bounds']
                    all_bounds.append(bounds)

                    rect = Rectangle(
                        bounds[:2, 0],
                        bounds[0, 1] - bounds[0, 0],
                        bounds[1, 1] - bounds[1, 0],
                        edgecolor='purple',
                        facecolor='none',
                        alpha=0.6,
                        linewidth=2,
                        linestyle=':',
                        label='Empirical t+1'
                    )
                    ax.add_patch(rect)

        # Draw tightest bounds for all timesteps (past, current, and lookahead)
        for timestep in sorted(tester.horizons.keys()):
            horizon = tester.horizons[timestep]
            bounds = horizon.get_tight_bound()

            if bounds is None:
                continue

            all_bounds.append(bounds)

            # Color based on timestep relative to current
            if timestep == 0:
                color = 'black'
                alpha = 0.7
                label = 'Initial (tightest)'
                linewidth = 3
            elif timestep < t:
                # Past timesteps - show tightest
                color = 'gray'
                alpha = 0.4
                label = f'Tightest t={timestep}' if timestep == t-1 else None
                linewidth = 2
            elif timestep == t:
                # Current timestep - use distinctive cyan/teal color
                color = 'cyan'
                alpha = 0.9
                label = f'Current t={t}'
                linewidth = 4
            elif timestep == t + 3:
                # Lookahead tightest
                color = 'red'
                alpha = 0.9
                label = f'Tightest t+3'
                linewidth = 3
            else:
                # Other future timesteps
                color = 'orange'
                alpha = 0.3
                label = None
                linewidth = 1.5

            # Draw filled rectangle for tightest bounds
            rect = Rectangle(
                bounds[:2, 0],
                bounds[0, 1] - bounds[0, 0],
                bounds[1, 1] - bounds[1, 0],
                edgecolor=color,
                facecolor=color,
                alpha=alpha * 0.3,
                linewidth=linewidth,
                label=label
            )
            ax.add_patch(rect)

        # Set axis limits
        if all_bounds:
            all_bounds_array = np.array(all_bounds)
            x_min = np.min(all_bounds_array[:, 0, 0])
            x_max = np.max(all_bounds_array[:, 0, 1])
            y_min = np.min(all_bounds_array[:, 1, 0])
            y_max = np.max(all_bounds_array[:, 1, 1])

            x_range = x_max - x_min
            y_range = y_max - y_min
            padding_x = x_range * 0.15
            padding_y = y_range * 0.15

            ax.set_xlim(x_min - padding_x, x_max + padding_x)
            ax.set_ylim(y_min - padding_y, y_max + padding_y)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        # Add text info
        info_text = f"Current t: {t}\n"
        info_text += f"Showing:\n"

        # Count calculation types at t+3
        if t + 3 in tester.horizons:
            h = tester.horizons[t + 3]
            concrete_count = sum(1 for c in h.calculations.values() if c['calc_type'] == CalculationType.CONCRETE)
            info_text += f"  • {concrete_count} Concrete steps (t→t+3)\n"
            info_text += f"  • Tightest vol @t+3: {h.get_tight_volume():.4f}\n"

        # Count empirical at t+1
        if t + 1 in tester.horizons:
            h = tester.horizons[t + 1]
            empirical_count = sum(1 for c in h.calculations.values() if c['calc_type'] == CalculationType.EMPIRICAL)
            if empirical_count > 0:
                info_text += f"  • {empirical_count} Empirical (t→t+1)\n"
                info_text += f"  • Tightest vol @t+1: {h.get_tight_volume():.4f}\n"

        info_text += f"\nTotal timesteps: {len(tester.horizons)}"

        ax.text(0.02, 0.98, info_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # Save frame
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, :3]  # Drop alpha channel to get RGB

        # Debug: print frame info
        print(f"  Frame {len(frames)}: shape={image.shape}, size={image.nbytes} bytes")

        frames.append(image)

        # Close this figure before moving to next
        plt.close(fig)

        # Move to next timestep
        t += 1

    # Save animation
    print("\n" + "="*80)
    print("SAVING ANIMATION")
    print("="*80)
    print(f"Total frames collected: {len(frames)}")

    if len(frames) == 0:
        print("ERROR: No frames were generated!")
        return

    # Print frame statistics
    frame_sizes = [f.nbytes for f in frames]
    print(f"Frame shape: {frames[0].shape}")
    print(f"Frame dtype: {frames[0].dtype}")
    print(f"Average frame size: {np.mean(frame_sizes)/1024:.1f} KB")
    print(f"Total data size: {sum(frame_sizes)/1024:.1f} KB")

    # Save as GIF using imageio (more reliable than matplotlib animation)
    import os
    try:
        import imageio
        use_imageio = True
        print("Using imageio for GIF creation")
    except ImportError:
        print("imageio not available, falling back to matplotlib animation")
        use_imageio = False

    output_dir = './animation_output'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'reachability_animation.gif')

    if use_imageio:
        # Use imageio for better GIF support with higher quality settings
        print(f"Saving {len(frames)} frames to GIF...")

        # Try with higher quality settings
        try:
            # quantizer=0 means no color quantization, subrectangles=False prevents optimization
            imageio.mimsave(output_file, frames, duration=500, loop=0, quantizer=0, subrectangles=False)
        except TypeError:
            # Fallback if quantizer parameter not supported
            print("  (using default quality settings)")
            imageio.mimsave(output_file, frames, duration=500, loop=0)

        # Check file size
        file_size = os.path.getsize(output_file)
        print(f" Animation saved to: {output_file}")
        print(f"  File size: {file_size/1024:.1f} KB ({file_size/1024/1024:.2f} MB)")

        # Also save as MP4 for better quality/compression
        mp4_file = os.path.join(output_dir, 'reachability_animation.mp4')
        print(f"\nAlso saving as MP4 for better quality...")
        try:
            imageio.mimsave(mp4_file, frames, fps=2, codec='libx264', quality=8)
            mp4_size = os.path.getsize(mp4_file)
            print(f" MP4 saved to: {mp4_file}")
            print(f"  File size: {mp4_size/1024:.1f} KB ({mp4_size/1024/1024:.2f} MB)")
        except Exception as e:
            print(f"  Could not create MP4: {e}")
            print(f"  (Try: pip install imageio-ffmpeg)")
    else:
        # Fallback to matplotlib
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_aspect('equal')

        im = ax.imshow(frames[0])
        ax.axis('off')

        def update(frame_num):
            im.set_data(frames[frame_num])
            return [im]

        anim = FuncAnimation(fig, update, frames=len(frames), interval=500, blit=False)

        writer = PillowWriter(fps=2)
        anim.save(output_file, writer=writer)
        print(f" Animation saved to: {output_file}")
        plt.close(fig)

    # Also save key frames as separate images
    # for i in [0, len(frames)//3, 2*len(frames)//3, len(frames)-1]:
    for i in range(len(frames)):
        if i < len(frames):
            frame_file = os.path.join(output_dir, f'frame_{i:03d}.png')
            plt.imsave(frame_file, frames[i])
            print(f" Frame {i} saved to: {frame_file}")

    print("\n Animation complete!")
    print(f"  Total frames: {len(frames)}")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--animate':
        animate()
    else:
        test1()
