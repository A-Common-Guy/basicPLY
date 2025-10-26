#!/usr/bin/env python3

import sys
from main import run_icp_and_save, load_and_visualize

def print_menu():
    print("\n" + "="*80)
    print("ICP Point Cloud Registration - Demo")
    print("="*80)
    print("\nStandard ICP:")
    print("  1. Run ICP WITHOUT visualization (faster)")
    print("  2. Run ICP WITH real-time visualization (slower)")
    print("\nICP with Automatic Mirroring Detection:")
    print("  3. Run ICP with AUTO-MIRRORING (no visualization, faster)")
    print("  4. Run ICP with AUTO-MIRRORING + visualization of best result")
    print("\nOther Options:")
    print("  5. Load and visualize previously saved results")
    print("  6. Exit")
    print("\nNote: Mirroring detection tests all 8 possible reflections (X, Y, Z axes)")
    print("      and automatically selects the best one based on convergence score.")
    print("      Colors: Red = source, Blue = target")
    

def main():
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            print("\n>>> Running standard ICP without visualization...")
            run_icp_and_save(visualize=False, use_mirroring=False)
            
        elif choice == '2':
            print("\n>>> Running standard ICP with real-time visualization...")
            print("Watch as the red point cloud aligns with the blue one!")
            run_icp_and_save(visualize=True, use_mirroring=False)
            
        elif choice == '3':
            print("\n>>> Running ICP with AUTOMATIC MIRRORING DETECTION...")
            print("This will try all 8 possible mirroring combinations and select the best one.")
            print("This may take a few minutes...")
            run_icp_and_save(visualize=False, use_mirroring=True)
            
        elif choice == '4':
            print("\n>>> Running ICP with AUTOMATIC MIRRORING DETECTION + visualization...")
            print("This will try all 8 possible mirroring combinations and show the best result.")
            print("This may take a few minutes...")
            run_icp_and_save(visualize=True, use_mirroring=True)
            
        elif choice == '5':
            print("\n>>> Loading saved results...")
            load_and_visualize()
            
        elif choice == '6':
            print("\nExiting...")
            sys.exit(0)
            
        else:
            print("\n⚠ Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()

