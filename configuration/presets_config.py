"""
ENCORE Analysis Presets Configuration Loader
=============================================

This module loads preset configurations from presets_config.jsonc.
All preset definitions and documentation are in the JSONC file.

Edit presets_config.jsonc to customize your analysis presets.
"""

import json
import re
from pathlib import Path


def _load_jsonc(filepath):
    """
    Load a JSONC file (JSON with comments).
    Strips // comments and /* */ block comments before parsing.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove single-line comments (// ...)
    content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
    
    # Remove multi-line comments (/* ... */)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    
    # Remove trailing commas before } or ]
    content = re.sub(r',\s*([}\]])', r'\1', content)
    
    return json.loads(content)


def _get_config_path():
    """Get the path to the config file"""
    return Path(__file__).parent / "presets_config.jsonc"


def _load_presets():
    """Load presets from the JSONC config file"""
    config_path = _get_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Presets config file not found: {config_path}")
    
    config = _load_jsonc(config_path)
    return config.get("presets", {})


# Cache the loaded presets
_PRESETS_CACHE = None


def _get_presets():
    """Get presets with caching"""
    global _PRESETS_CACHE
    if _PRESETS_CACHE is None:
        _PRESETS_CACHE = _load_presets()
    return _PRESETS_CACHE


def reload_presets():
    """Force reload presets from file (call after editing config)"""
    global _PRESETS_CACHE
    _PRESETS_CACHE = None
    return _get_presets()


# =============================================================================
# PUBLIC API
# =============================================================================

def get_preset_names():
    """Return list of all preset names"""
    return list(_get_presets().keys())


def get_preset_params(preset_name):
    """
    Get the parameter tuple for a preset.
    
    Returns:
        tuple: (intensity, timescale, direct_indirect, spatial, comp_svc_rating, svc_act_intensity)
        None: if preset not found
    """
    presets = _get_presets()
    if preset_name in presets:
        params = presets[preset_name]["params"]
        return (
            params["intensity"],
            params["timescale"],
            params["direct_indirect"],
            params["spatial"],
            params["comp_svc_rating"],
            params["svc_act_intensity"]
        )
    return None


def get_preset_description(preset_name):
    """Get the description for a preset"""
    presets = _get_presets()
    if preset_name in presets:
        return presets[preset_name].get("description", "")
    return None


def get_preset_use_case(preset_name):
    """Get the use case for a preset"""
    presets = _get_presets()
    if preset_name in presets:
        return presets[preset_name].get("use_case", "")
    return None


def get_all_presets():
    """Return the full presets dictionary"""
    return _get_presets()


def get_presets_as_param_dict():
    """
    Return presets as a simple name -> params tuple dictionary.
    Useful for backwards compatibility with existing code.
    
    Returns:
        dict: {"preset_name": (params_tuple), ...}
    """
    result = {}
    for name in get_preset_names():
        result[name] = get_preset_params(name)
    return result


def get_default_preset_name():
    """Return the default/recommended preset name"""
    # Look for 'Balanced' as the default
    names = get_preset_names()
    for name in names:
        if 'balanced' in name.lower():
            return name
    # Fallback to first preset
    return names[0] if names else None


def get_preset_category(preset_name):
    """Get the category for a preset"""
    presets = _get_presets()
    if preset_name in presets:
        return presets[preset_name].get("category", "Other")
    return None


def get_preset_categories():
    """
    Return list of unique category names in order they appear.
    
    Returns:
        list: ["Scope", "Risk", "Timeline", "Spatial", ...]
    """
    presets = _get_presets()
    categories = []
    for config in presets.values():
        cat = config.get("category", "Other")
        if cat not in categories:
            categories.append(cat)
    return categories


def get_presets_by_category():
    """
    Return presets grouped by category.
    
    Returns:
        dict: {
            "Scope": ["Conservative (High Confidence)", "Balanced", ...],
            "Risk": ["High-Risk Pathways", ...],
            ...
        }
    """
    presets = _get_presets()
    by_category = {}
    
    for name, config in presets.items():
        cat = config.get("category", "Other")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(name)
    
    return by_category


def get_category_descriptions():
    """
    Return descriptions for each category.
    
    Returns:
        dict: {"Scope": "Control how wide/narrow the analysis is", ...}
    """
    return {
        "Scope": "Control how wide/narrow the analysis is",
        "Risk": "For risk assessment and materiality analysis",
        "Timeline": "When do impacts occur?",
        "Spatial": "Where do impacts occur?"
    }


def print_preset_summary():
    """Print a formatted summary of all presets"""
    presets = _get_presets()
    
    print("=" * 80)
    print("AVAILABLE ANALYSIS PRESETS")
    print("=" * 80)
    
    for name, config in presets.items():
        params = config["params"]
        print(f"\n{name}")
        print("-" * len(name))
        print(f"Parameters:")
        print(f"  - Activity -> Pressures intensity: {params['intensity']}+")
        print(f"  - Timescale: {params['timescale']}")
        print(f"  - Direct/Indirect: {params['direct_indirect']}")
        print(f"  - Spatial: {params['spatial']}")
        print(f"  - Component -> Service sensitivity: {params['comp_svc_rating']}")
        print(f"  - Service -> Activity dependency: {params['svc_act_intensity']}+")
        print(f"\nDescription: {config.get('description', 'N/A')}")
        print(f"\nUse case: {config.get('use_case', 'N/A')}")
        print()


def print_preset_table():
    """Print presets in a compact table format"""
    presets = _get_presets()
    
    print("=" * 100)
    print("PRESET COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Preset Name':<35} {'Int':<4} {'Time':<6} {'Dir':<8} {'Spat':<8} {'Sens':<5} {'Dep':<4}")
    print("-" * 100)
    
    for name, config in presets.items():
        p = config["params"]
        short_name = name[:33] + ".." if len(name) > 35 else name
        print(f"{short_name:<35} {p['intensity']:<4} {p['timescale']:<6} {p['direct_indirect']:<8} "
              f"{p['spatial']:<8} {p['comp_svc_rating']:<5} {p['svc_act_intensity']:<4}")
    
    print("=" * 100)
    print("\nLegend: Int=Intensity, Time=Timescale, Dir=Direct/Indirect, Spat=Spatial, Sens=Sensitivity, Dep=Dependency")


if __name__ == "__main__":
    # If run directly, print preset summary
    print_preset_table()
    print("\n")
    print_preset_summary()
