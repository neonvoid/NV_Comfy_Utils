#!/usr/bin/env python3
"""
Manual VAE LUT Creator - Standalone Script
==========================================

Use this script if you prefer to create VAE LUTs outside of ComfyUI,
or if you already have diagnostic statistics from multiple test runs.

Usage:
    python create_vae_lut_manual.py

Follow the prompts to input your test data.
"""

import json
from datetime import datetime
from pathlib import Path


def print_header():
    print("=" * 70)
    print("VAE LUT MANUAL CREATOR")
    print("=" * 70)
    print()
    print("This script helps you create a VAE correction LUT from diagnostic data.")
    print("You'll need statistics from your Video Extension Diagnostic runs.")
    print()


def get_vae_info():
    print("Step 1: VAE Information")
    print("-" * 70)
    vae_name = input("VAE name (e.g., 'sdxl_vae'): ").strip()
    return vae_name


def get_original_stats():
    print("\nStep 2: Original Video Statistics")
    print("-" * 70)
    print("Enter stats from your ORIGINAL video (0 VAE cycles):")
    
    mean = float(input("  Mean (e.g., 0.413611): "))
    std = float(input("  Std (e.g., 0.336798): "))
    min_val = float(input("  Min (e.g., 0.000000): "))
    max_val = float(input("  Max (e.g., 1.000000): "))
    clip_0 = float(input("  % clipped to 0 (e.g., 3.15): ")) / 100
    clip_1 = float(input("  % clipped to 1 (e.g., 0.01): ")) / 100
    
    return {
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "clip_0": clip_0,
        "clip_1": clip_1,
    }


def get_cycle_data(orig_stats):
    print("\nStep 3: Cycle Video Statistics")
    print("-" * 70)
    print("Enter stats for each VAE cycle count you tested.")
    print("(Enter blank line when done)")
    print()
    
    cycle_data = {}
    
    while True:
        cycle_input = input(f"\nCycle count (e.g., 1, 2, 3...) or [DONE]: ").strip()
        
        if not cycle_input or cycle_input.lower() == "done":
            break
        
        try:
            cycle_num = int(cycle_input)
        except ValueError:
            print("Invalid number, try again.")
            continue
        
        print(f"  Stats for {cycle_num} VAE cycles:")
        mean = float(input("    Mean: "))
        std = float(input("    Std: "))
        clip_0 = float(input("    % clipped to 0: ")) / 100
        
        # Calculate drift
        mean_drift = mean - orig_stats["mean"]
        mean_drift_pct = (mean_drift / orig_stats["mean"]) * 100
        std_drift = std - orig_stats["std"]
        std_drift_pct = (std_drift / orig_stats["std"]) * 100
        
        # Calculate corrections
        brightness_correction = orig_stats["mean"] / mean if mean > 0 else 1.0
        contrast_correction = orig_stats["std"] / std if std > 0 else 1.0
        
        # Shadow lift
        clip_increase = clip_0 - orig_stats["clip_0"]
        shadow_lift = max(0.0, clip_increase * 0.05)
        
        cycle_data[cycle_num] = {
            "stats": {
                "mean": mean,
                "std": std,
                "clip_0": clip_0,
            },
            "corrections": {
                "brightness_mult": brightness_correction,
                "contrast_mult": contrast_correction,
                "shadow_lift": shadow_lift,
            },
            "drift": {
                "mean_drift": mean_drift,
                "mean_drift_pct": mean_drift_pct,
                "std_drift": std_drift,
                "std_drift_pct": std_drift_pct,
            }
        }
        
        print(f"    → Brightness correction: ×{brightness_correction:.4f}")
        print(f"    → Contrast correction:   ×{contrast_correction:.4f}")
        print(f"    → Shadow lift:           +{shadow_lift:.6f}")
        print(f"    → Mean drift: {mean_drift_pct:+.2f}%")
    
    return cycle_data


def generate_lut(vae_name, orig_stats, cycle_data):
    lut = {
        "vae_name": vae_name,
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "calibration_video_stats": orig_stats,
        "corrections": {},
        "drift_analysis": {},
    }
    
    # Build corrections dict
    drift_rates = []
    for cycle_num, data in sorted(cycle_data.items()):
        lut["corrections"][cycle_num] = {
            "brightness_mult": round(data["corrections"]["brightness_mult"], 6),
            "contrast_mult": round(data["corrections"]["contrast_mult"], 6),
            "shadow_lift": round(data["corrections"]["shadow_lift"], 6),
        }
        
        lut["drift_analysis"][cycle_num] = {
            "mean_drift": round(data["drift"]["mean_drift"], 6),
            "mean_drift_pct": round(data["drift"]["mean_drift_pct"], 2),
            "std_drift": round(data["drift"]["std_drift"], 6),
            "std_drift_pct": round(data["drift"]["std_drift_pct"], 2),
        }
        
        drift_rates.append(data["drift"]["mean_drift_pct"] / cycle_num)
    
    # Calculate drift model
    if drift_rates:
        avg_drift_rate = sum(drift_rates) / len(drift_rates)
        lut["drift_model"] = {
            "type": "linear",
            "mean_decay_rate_per_cycle": round(abs(avg_drift_rate) / 100, 6),
        }
    
    return lut


def print_report(lut):
    print("\n" + "=" * 70)
    print("GENERATED LUT REPORT")
    print("=" * 70)
    print(f"VAE: {lut['vae_name']}")
    print(f"Date: {lut['generation_date']}")
    print(f"\nOriginal Stats:")
    print(f"  Mean: {lut['calibration_video_stats']['mean']:.6f}")
    print(f"  Std:  {lut['calibration_video_stats']['std']:.6f}")
    print(f"\nCycle Corrections:")
    
    for cycle_num in sorted([int(k) for k in lut['corrections'].keys()]):
        corr = lut['corrections'][cycle_num]
        drift = lut['drift_analysis'][cycle_num]
        print(f"\n  Cycle {cycle_num}:")
        print(f"    Drift: {drift['mean_drift_pct']:+.2f}%")
        print(f"    Brightness: ×{corr['brightness_mult']:.4f}")
        print(f"    Contrast:   ×{corr['contrast_mult']:.4f}")
        print(f"    Shadow Lift: +{corr['shadow_lift']:.6f}")
    
    if "drift_model" in lut:
        print(f"\nDrift Model:")
        print(f"  Type: {lut['drift_model']['type']}")
        print(f"  Decay rate: {lut['drift_model']['mean_decay_rate_per_cycle']:.6f} per cycle")
    
    print("=" * 70)


def save_lut(lut, vae_name):
    # Sanitize filename
    safe_name = "".join(c for c in vae_name if c.isalnum() or c in "._- ")
    filename = f"vae_lut_{safe_name}.json"
    
    print(f"\nStep 4: Save LUT")
    print("-" * 70)
    save_path = input(f"Save as [{filename}]: ").strip()
    
    if not save_path:
        save_path = filename
    
    # Create output directory if needed
    output_dir = Path(__file__).parent / "luts"
    output_dir.mkdir(exist_ok=True)
    
    full_path = output_dir / save_path
    
    with open(full_path, 'w') as f:
        json.dump(lut, f, indent=2)
    
    print(f"\n✓ LUT saved to: {full_path}")
    print(f"\nYou can now use this LUT with the 'VAE Correction Applier' node.")
    print(f"Copy the contents of {full_path} into the lut_json parameter.")


def main():
    print_header()
    
    try:
        vae_name = get_vae_info()
        orig_stats = get_original_stats()
        cycle_data = get_cycle_data(orig_stats)
        
        if not cycle_data:
            print("\nError: No cycle data entered. Need at least 1 cycle.")
            return
        
        lut = generate_lut(vae_name, orig_stats, cycle_data)
        print_report(lut)
        save_lut(lut, vae_name)
        
        print("\n✅ LUT creation complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

