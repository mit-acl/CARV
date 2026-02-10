
from REAL_integrated_sim import setup_analyzer, setup_backward_analyzer, ReachabilityTester, CalculationType, Obstacles
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import numpy as np

def test():
# [[2.5, 3.0], [-0.25, 0.25]]
    obs1 = np.array([[-5.0, -3.0], [2.0, 4.0]])
    obs2 = np.array([[0.0, 1.6], [-1.0, -0.4]])
    obstacles = Obstacles([obs1, obs2])

# [ 1.5735503,  2.4511724],
#        [-0.9427511, -0.7375897]

    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')

    print("\nCreating interactive tester...")
    tester = ReachabilityTester(analyzer, obstacles_list=obstacles)

    print("\n" + "=" * 20 + " Initial State " + "=" * 20 +"")
    print(tester.horizons[0])
    t = 0

    print(tester.symbolic(0,7))

def test1():
    analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')
    # analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')


    print("\nCreating interactive tester...")
    tester = ReachabilityTester(analyzer)

    print("\n" + "=" * 20 + " Initial State " + "=" * 20)
    print(tester.horizons[0])
    t = 0

    print(tester.symbolic(0,7))

def test2():
    analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')
    # analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')

    print("\nCreating interactive tester...")
    tester = ReachabilityTester(analyzer)

    print("\n" + "=" * 20 + " Initial State " + "=" * 20)
    print(tester.horizons[0])
    t = 0
    tester.symbolic(0,10)
    tester.concrete(0,10)

# def test_backward():
#     """Test backward reachability with animation"""
#     # analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
#     analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
#     backward_analyzer = setup_backward_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
#     tester = ReachabilityTester(analyzer, backward_analyzer=backward_analyzer)
#     final_state_range = np.array([[4.5, 5.0], [-0.25, 0.25]])
#     tester.backward(final_state_range)
#     # print("\nCreating tester with backward reachability...")
#     # tester = ReachabilityTester(analyzer)

#     # # Forward propagation to t=5
#     # print("\n" + "=" * 20 + " Forward to t=5 " + "=" * 20)
#     # for t in range(5):
#     #     tester.concrete(t, t+1)
#     #     tester.real_state_empirical(t, t+1)
    
#     # print("\n" + "=" * 20 + " Target Set at t=5 " + "=" * 20)
#     # tester.horizons[5].list_calculations()
    
#     # # Backward propagation from t=5 to t=0, t=1, t=2, t=3, t=4
#     # print("\n" + "=" * 20 + " Backward Propagation " + "=" * 20)
#     # for start_t in [0, 1, 2, 3, 4]:
#     #     tester.backward(target_timestep=5, start_timestep=start_t)
#     #     tester.horizons[start_t].list_calculations()
    
#     # print("\n" + "=" * 20 + " Summary " + "=" * 20)
#     # for t in range(6):
#     #     print(f"t={t}: {tester.horizons[t]}")

def test_backward():
    """Test backward reachability with proper horizon integration"""
    print("=" * 80)
    print("TESTING BACKWARD REACHABILITY")
    print("=" * 80)
    
    # Setup analyzers
    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
    backward_analyzer = setup_backward_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
    
    # Create tester with both analyzers
    tester = ReachabilityTester(analyzer, backward_analyzer=backward_analyzer)
    
    print("\n" + "=" * 60)
    print("PHASE 1: FORWARD PROPAGATION")
    print("=" * 60)
    
    # Forward propagation to build up some timesteps
    print("\nPropagating forward from t=0 to t=10...")
    for t in range(10):
        tester.concrete(t, t+1)
        tester.real_state_empirical(t, t+1)
        print(f"  t={t} → t={t+1}: tight vol = {tester.horizons[t+1].get_tight_volume():.6f}")
    
    print("\n" + "=" * 60)
    print("PHASE 2: BACKWARD PROPAGATION")
    print("=" * 60)
    
    # Test backward from t=10, going back 5 timesteps
    start_timestep = 10
    num_steps = 5
    
    print(f"\nBackward propagation: from t={start_timestep}, going back {num_steps} steps")
    print(f"This should create backward sets at t={start_timestep - num_steps} to t={start_timestep - 1}")
    
    tester.backward(start_timestep=start_timestep, num_steps=num_steps)
    
    print("\n" + "=" * 60)
    print("PHASE 3: VERIFY RESULTS")
    print("=" * 60)
    
    # Check the results at each timestep
    print(f"\nChecking timesteps {start_timestep - num_steps} to {start_timestep}:\n")
    
    for t in range(start_timestep - num_steps, start_timestep + 1):
        if t in tester.horizons:
            horizon = tester.horizons[t]
            
            # Count calculation types
            concrete_count = sum(1 for c in horizon.calculations.values() 
                               if c['calc_type'] == CalculationType.CONCRETE)
            empirical_count = sum(1 for c in horizon.calculations.values() 
                                if c['calc_type'] == CalculationType.EMPIRICAL)
            backward_count = sum(1 for c in horizon.calculations.values() 
                               if c['calc_type'] == CalculationType.BACKWARD)
            
            print(f"t={t}:")
            print(f"  Concrete: {concrete_count}, Empirical: {empirical_count}, Backward: {backward_count}")
            print(f"  Tight volume: {horizon.get_tight_volume():.6f}")
            
            # Show backward calculation details if present
            for calc_id, calc_info in horizon.calculations.items():
                if calc_info['calc_type'] == CalculationType.BACKWARD:
                    print(f"    → Backward set: origin=t{calc_info['origin_timestep']}, "
                          f"steps_to_target={calc_info['step_size']}, "
                          f"volume={calc_info['volume']:.6f}")
            print()
    
    print("=" * 60)
    print("PHASE 4: DETAILED LISTINGS")
    print("=" * 60)
    
    # Show detailed calculations for a few key timesteps
    for t in [start_timestep - num_steps, start_timestep - 3, start_timestep - 1, start_timestep]:
        if t in tester.horizons:
            print(f"\n--- Detailed view of t={t} ---")
            tester.horizons[t].list_calculations()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    # Summary
    print("\nSummary:")
    print(f"  Forward timesteps: 0 to {start_timestep}")
    print(f"  Backward from: t={start_timestep}")
    print(f"  Backward steps: {num_steps}")
    print(f"  Backward sets at: t={start_timestep - num_steps} to t={start_timestep - 1}")
    
    total_backward = sum(
        1 for h in tester.horizons.values()
        for c in h.calculations.values()
        if c['calc_type'] == CalculationType.BACKWARD
    )
    print(f"  Total backward calculations: {total_backward}")
    
    return tester

def animate():
    """Create animation of reachability propagation following test pattern"""
    print("=" * 80)
    print("CREATING ANIMATION")
    print("=" * 80)

    # analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')
    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')


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

# def animate_backward():
#     """Animate backward reachability using updated backward() API"""
#     print("=" * 80)
#     print("CREATING BACKWARD REACHABILITY ANIMATION")
#     print("=" * 80)

#     analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
#     backward_analyzer = setup_backward_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')

#     tester = ReachabilityTester(analyzer, backward_analyzer=backward_analyzer)

#     # --------------------------------------------------
#     # PHASE 1: Forward propagation (build horizons)
#     # --------------------------------------------------
#     T = 10
#     for t in range(T):
#         tester.concrete(t, t + 1)
#         tester.real_state_empirical(t, t + 1)

#     if T not in tester.horizons:
#         raise RuntimeError("No horizon at target timestep")

#     frames = []
#     max_back_steps = 6

#     print("\nGenerating backward animation frames...")

#     for k in range(1, max_back_steps + 1):
#         print(f"  Backward step {k}/{max_back_steps}")

#         tester.backward(start_timestep=T, num_steps=k)

#         fig, ax = plt.subplots(figsize=(10, 7))
#         ax.set_title(
#             f'Backward Reachability (target t={T}, depth={k})',
#             fontsize=14, fontweight='bold'
#         )
#         ax.set_xlabel('State 1')
#         ax.set_ylabel('State 2')
#         ax.grid(True, alpha=0.3)

#         all_bounds = []

#         # --------------------------------------------------
#         # Draw target tight bound at T
#         # --------------------------------------------------
#         target_bounds = tester.horizons[T].get_tight_bound()
#         if target_bounds is not None:
#             rect = Rectangle(
#                 target_bounds[:2, 0],
#                 target_bounds[0, 1] - target_bounds[0, 0],
#                 target_bounds[1, 1] - target_bounds[1, 0],
#                 edgecolor='red',
#                 facecolor='red',
#                 alpha=0.35,
#                 linewidth=3,
#                 label='Target (tight bound @ T)'
#             )
#             ax.add_patch(rect)
#             all_bounds.append(target_bounds)

#         # --------------------------------------------------
#         # Draw backward reachable sets
#         # --------------------------------------------------
#         for t in range(T - k, T):
#             if t not in tester.horizons:
#                 continue

#             horizon = tester.horizons[t]
#             for calc in horizon.calculations.values():
#                 if calc['calc_type'] != CalculationType.BACKWARD:
#                     continue
#                 if calc['origin_timestep'] != T:
#                     continue

#                 bounds = calc['bounds']
#                 all_bounds.append(bounds)

#                 rect = Rectangle(
#                     bounds[:2, 0],
#                     bounds[0, 1] - bounds[0, 0],
#                     bounds[1, 1] - bounds[1, 0],
#                     edgecolor='blue',
#                     facecolor='none',
#                     linewidth=2,
#                     linestyle='--',
#                     alpha=0.8,
#                     label=f'Backward @ t={t}' if t == T - k else None
#                 )
#                 ax.add_patch(rect)

#         # --------------------------------------------------
#         # Forward tight bounds (context)
#         # --------------------------------------------------
#         for t, horizon in tester.horizons.items():
#             if t > T:
#                 continue

#             bounds = horizon.get_tight_bound()
#             if bounds is None:
#                 continue

#             all_bounds.append(bounds)
#             rect = Rectangle(
#                 bounds[:2, 0],
#                 bounds[0, 1] - bounds[0, 0],
#                 bounds[1, 1] - bounds[1, 0],
#                 edgecolor='black',
#                 facecolor='black',
#                 alpha=0.15,
#                 linewidth=1,
#                 label='Forward tight bounds' if t == 0 else None
#             )
#             ax.add_patch(rect)

#         # --------------------------------------------------
#         # Axis limits
#         # --------------------------------------------------
#         if all_bounds:
#             B = np.array(all_bounds)
#             x_min, x_max = B[:, 0, 0].min(), B[:, 0, 1].max()
#             y_min, y_max = B[:, 1, 0].min(), B[:, 1, 1].max()

#             pad_x = 0.2 * (x_max - x_min)
#             pad_y = 0.2 * (y_max - y_min)

#             ax.set_xlim(x_min - pad_x, x_max + pad_x)
#             ax.set_ylim(y_min - pad_y, y_max + pad_y)

#         # Legend (deduplicated)
#         handles, labels = ax.get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         ax.legend(by_label.values(), by_label.keys(), loc='upper left')

#         # Info box
#         ax.text(
#             0.02, 0.02,
#             f"Target timestep: T={T}\nBackward depth: {k}\nShowing t={T-k} → t={T-1}",
#             transform=ax.transAxes,
#             fontsize=9,
#             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
#         )

#         # --------------------------------------------------
#         # Capture frame
#         # --------------------------------------------------
#         fig.canvas.draw()
#         image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
#         image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
#         frames.append(image[:, :, :3])

#         plt.close(fig)

#     # --------------------------------------------------
#     # Save animation
#     # --------------------------------------------------
#     import os, imageio

#     output_dir = './animation_output'
#     os.makedirs(output_dir, exist_ok=True)
#     gif_path = os.path.join(output_dir, 'backward_reachability.gif')

#     durations = [1000] * len(frames)  # 5000 ms per frame
#     imageio.mimsave(gif_path, frames, duration=durations, loop=0)

#     print("\n" + "=" * 80)
#     print("BACKWARD ANIMATION COMPLETE")
#     print("=" * 80)
#     print(f"Saved to: {gif_path}")

def plot_backward_debug():
    """Static debug plot: forward empirical reachability + backward reachable sets"""

    print("=" * 80)
    print("PLOTTING BACKWARD REACHABILITY DEBUG VIEW")
    print("=" * 80)

    import os
    output_dir = './animation_output'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'backward_reachability_debug.png')

    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
    backward_analyzer = setup_backward_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')

    tester = ReachabilityTester(analyzer, backward_analyzer=backward_analyzer)

    # --------------------------------------------------
    # Forward propagation (empirical only)
    # --------------------------------------------------
    T = 10
    for t in range(T):
        tester.real_state_empirical(t, t + 1)  # only empirical propagation

    # --------------------------------------------------
    # Backward propagation (single call)
    # --------------------------------------------------
    BACKWARD_STEPS = 6
    tester.backward(start_timestep=T, num_steps=BACKWARD_STEPS)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Backward Reachability Debug Plot", fontsize=14, fontweight='bold')
    ax.set_xlabel("State 1")
    ax.set_ylabel("State 2")
    ax.grid(True, alpha=0.3)

    all_bounds = []

    # --------------------------------------------------
    # Forward empirical reachable sets (history)
    # --------------------------------------------------
    for t in range(T + 1):
        horizon = tester.horizons.get(t)
        if horizon is None:
            continue

        # Only include EMPIRICAL calculations
        empirical_bounds_list = [
            calc['bounds']
            for calc in horizon.calculations.values()
            if calc['calc_type'] == CalculationType.EMPIRICAL
        ]

        for bounds in empirical_bounds_list:
            all_bounds.append(bounds)
            rect = Rectangle(
                bounds[:2, 0],
                bounds[0, 1] - bounds[0, 0],
                bounds[1, 1] - bounds[1, 0],
                edgecolor='black',
                facecolor='none',
                linewidth=1,
                alpha=0.6,
                label='Forward empirical' if t == 0 else None
            )
            ax.add_patch(rect)

    # --------------------------------------------------
    # Target tight set at T
    # --------------------------------------------------
    target_bounds_list = [
        calc['bounds']
        for calc in tester.horizons[T].calculations.values()
        if calc['calc_type'] == CalculationType.EMPIRICAL
    ]
    if target_bounds_list:
        # Take the first one (or merge if multiple)
        target_bounds = target_bounds_list[0]
        all_bounds.append(target_bounds)
        rect = Rectangle(
            target_bounds[:2, 0],
            target_bounds[0, 1] - target_bounds[0, 0],
            target_bounds[1, 1] - target_bounds[1, 0],
            edgecolor='red',
            facecolor='red',
            alpha=0.35,
            linewidth=3,
            label='Target set (empirical @ T)'
        )
        ax.add_patch(rect)

    # --------------------------------------------------
    # Backward reachable sets
    # --------------------------------------------------
    for t in range(T - BACKWARD_STEPS, T):
        horizon = tester.horizons.get(t)
        if horizon is None:
            continue

        for calc in horizon.calculations.values():
            if calc['calc_type'] != CalculationType.BACKWARD:
                continue
            if calc['origin_timestep'] != T:
                continue

            bounds = calc['bounds']
            all_bounds.append(bounds)

            rect = Rectangle(
                bounds[:2, 0],
                bounds[0, 1] - bounds[0, 0],
                bounds[1, 1] - bounds[1, 0],
                edgecolor='blue',
                facecolor='none',
                linestyle='--',
                linewidth=2,
                alpha=0.9,
                label='Backward reachable sets' if t == T - BACKWARD_STEPS else None
            )
            ax.add_patch(rect)

    # --------------------------------------------------
    # Axis limits
    # --------------------------------------------------
    if all_bounds:
        B = np.array(all_bounds)
        x_min, x_max = B[:, 0, 0].min(), B[:, 0, 1].max()
        y_min, y_max = B[:, 1, 0].min(), B[:, 1, 1].max()

        pad_x = 0.2 * (x_max - x_min)
        pad_y = 0.2 * (y_max - y_min)

        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    # --------------------------------------------------
    # Save figure to animation_output folder
    # --------------------------------------------------
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved backward reachability debug plot to: {output_file}")

    plt.close(fig)



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--animate':
        animate()
    elif len(sys.argv) > 1 and sys.argv[1] == '--backward': 
        test_backward()
    elif len(sys.argv) > 1 and sys.argv[1] == '--animate-backward':  
        animate_backward()    
    elif len(sys.argv) > 1 and sys.argv[1] == '--backward-debug':  
        plot_backward_debug()                                       
    else:
        test1()