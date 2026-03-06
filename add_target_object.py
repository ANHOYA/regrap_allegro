#!/usr/bin/env python3
# python add_target_object.py assets/ycb/새로운_물체_폴더명
import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python add_target_object.py <ycb_folder_name_or_path>")
        print("Example: python add_target_object.py 055_baseball")
        sys.exit(1)
        
    target = sys.argv[1]
    folder_name = os.path.basename(os.path.normpath(target))
    
    # Auto-generate key and label (e.g. "055_baseball" -> "baseball", "Baseball")
    parts = folder_name.split('_', 1)
    if len(parts) == 2 and parts[0].isdigit():
        key_name = parts[1]
    else:
        key_name = folder_name
        
    label_name = key_name.replace('_', ' ').title()
    
    # Physics default properties
    mass = 0.15  # Default mass (150g is good for a baseball)
    pos = "(0.05, 0.5, 0.0)"
    
    config_path = os.path.join(os.path.dirname(__file__), "target_object.py")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        sys.exit(1)
        
    # Find bounds of YCB_OBJECTS
    start_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("YCB_OBJECTS = {"):
            start_idx = i
            break
            
    if start_idx == -1:
        print("Error: Could not find YCB_OBJECTS dictionary in target_object.py")
        sys.exit(1)
        
    end_idx = -1
    open_braces = 0
    for i in range(start_idx, len(lines)):
        open_braces += lines[i].count('{')
        open_braces -= lines[i].count('}')
        if open_braces == 0 and i > start_idx:
            end_idx = i
            break
            
    if end_idx == -1:
        print("Error: Could not find end of YCB_OBJECTS dictionary")
        sys.exit(1)
        
    # Check if already exists
    exists = False
    for i in range(start_idx, end_idx):
        if f'"{key_name}":' in lines[i] or f"'{key_name}':" in lines[i]:
            exists = True
            break
            
    if not exists:
        new_entry = f"""    "{key_name}": {{
        "folder": "{folder_name}",
        "mass": {mass},
        "position": {pos},
        "label": "{label_name}",
        "static_friction": 0.8,
        "dynamic_friction": 0.6,
        "restitution": 0.3,
    }},\n"""
        lines.insert(end_idx, new_entry)
        print(f"✅ Added '{key_name}' entry to YCB_OBJECTS")
    else:
        print(f"ℹ️ '{key_name}' already exists in YCB_OBJECTS")
        
    # Switch ACTIVE_OBJECT
    for i, line in enumerate(lines):
        if line.startswith("ACTIVE_OBJECT = "):
            lines[i] = f'ACTIVE_OBJECT = "{key_name}"\n'
            print(f"✅ Set ACTIVE_OBJECT = \"{key_name}\"")
            break
            
    with open(config_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
        
    print(f"🎉 Successfully configured '{key_name}' in target_object.py!")
    print(f"   (Run main.py and Isaac Sim will automatically convert its OBJ to USD layout)")

if __name__ == "__main__":
    main()
