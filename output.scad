scad
// Quadcopter Propeller - Parametric Design
// A fully customizable propeller for quadcopters with aerodynamic optimization

// ======== MAIN PARAMETERS ========
// Basic dimensions
prop_diameter = 127;    // Propeller diameter in mm (5 inches)
hub_diameter = 12;      // Hub diameter in mm
hub_height = 7;         // Hub height in mm
shaft_diameter = 5;     // Motor shaft diameter in mm (M5)
num_blades = 2;         // Number of blades (2 is standard for most quads)

// Blade parameters
blade_chord_max = 14;   // Maximum width of blade in mm
blade_chord_tip = 5;    // Width at blade tip in mm
blade_thickness = 2.2;  // Maximum thickness at blade root in mm
pitch_angle = 12;       // Base pitch angle (degrees)
twist_angle = 15;       // Additional twist from root to tip (degrees)
airfoil_camber = 0.04;  // Airfoil camber ratio (0.0-0.1 recommended)

// Performance tuning
tip_taper_factor = 0.7; // Controls how blade width decreases toward tip (0.5-1.0)
efficiency_factor = 1.2;// Multiplier for airfoil lift curve (1.0-1.5)

// Resolution settings
$fn = 80;               // Overall resolution for curved surfaces
blade_segments = 20;    // Number of segments along blade length
profile_resolution = 24;// Number of points in airfoil cross-section

// ======== DERIVED VARIABLES ========
blade_length = (prop_diameter - hub_diameter) / 2;
blade_root_offset = hub_diameter / 2;

// ======== MODULES ========

// Generate an airfoil cross-section profile
module airfoil_profile(chord, thickness, camber, angle) {
    // Create upper and lower curves of the airfoil
    upper_points = [for (i = [0:profile_resolution]) 
        let(
            t = i / profile_resolution,
            x = chord * (1 - cos(t * 180)),
            // NACA-inspired thickness distribution
            thick = thickness * (0.2969 * sqrt(x/chord) - 
                    0.1260 * (x/chord) - 0.3516 * pow(x/chord, 2) + 
                    0.2843 * pow(x/chord, 3) - 0.1015 * pow(x/chord, 4)),
            // Camber line
            yc = camber * efficiency_factor * chord * sin(t * 180)
        )
        [x, yc + thick]
    ];
    
    lower_points = [for (i = [profile_resolution:-1:0]) 
        let(
            t = i / profile_resolution,
            x = chord * (1 - cos(t * 180)),
            // NACA-inspired thickness distribution
            thick = thickness * (0.2969 * sqrt(x/chord) - 
                    0.1260 * (x/chord) - 0.3516 * pow(x/chord, 2) + 
                    0.2843 * pow(x/chord, 3) - 0.1015 * pow(x/chord, 4)),
            // Camber line
            yc = camber * efficiency_factor * chord * sin(t * 180)
        )
        [x, yc - thick]
    ];
    
    // Combine upper and lower curves and rotate to the specified angle
    rotate([0, 0, angle])
        polygon(points = concat(upper_points, lower_points));
}

// Generate a single propeller blade
module blade() {
    // Create blade segments with varying properties from root to tip
    for (i = [0:blade_segments-1]) {
        // Position along blade (normalized 0-1)
        t = i / blade_segments;
        z1 = t * blade_length;
        z2 = (i + 1) / blade_segments * blade_length;
        
        // Calculate chord length at each position with non-linear taper
        chord1 = blade_chord_max * (1 - t * (1 - blade_chord_tip/blade_chord_max) * pow(t, tip_taper_factor));
        chord2 = blade_chord_max * (1 - (i+1)/blade_segments * (1 - blade_chord_tip/blade_chord_max) * 
                 pow((i+1)/blade_segments, tip_taper_factor));
        
        // Calculate thickness (thinner toward tip for better dynamics)
        thickness1 = blade_thickness * (1 - 0.6 * t);
        thickness2 = blade_thickness * (1 - 0.6 * ((i+1)/blade_segments));
        
        // Calculate twist angle (non-linear distribution for optimal thrust)
        angle1 = pitch_angle + twist_angle * (1 - pow(1 - t, 1.5));
        angle2 = pitch_angle + twist_angle * (1 - pow(1 - (i+1)/blade_segments, 1.5));
        
        // Create segment by connecting profiles at z1 and z2
        hull() {
            translate([blade_root_offset + z1, 0, 0])
                linear_extrude(height = 0.01)
                    airfoil_profile(chord1, thickness1, airfoil_camber, angle1);
            
            translate([blade_root_offset + z2, 0, 0])
                linear_extrude(height = 0.01)
                    airfoil_profile(chord2, thickness2, airfoil_camber, angle2);
        }
    }
}

// Hub with motor shaft mounting hole and set screw hole
module hub() {
    difference() {
        union() {
            // Main hub cylinder
            cylinder(h = hub_height, d = hub_diameter, center = true);
            
            // Hub reinforcement at base
            translate([0, 0, -hub_height/2])
                cylinder(h = hub_height/4, d1 = hub_diameter * 1.2, d2 = hub_diameter, center = false);
            
            // Top cap with aerodynamic shape
            translate([0, 0, hub_height/2 - 0.01])
                cylinder(h = hub_height/3, d1 = hub_diameter, d2 = hub_diameter * 0.6, center = false);
        }
        
        // Shaft hole
        cylinder(h = hub_height * 1.5, d = shaft_diameter, center = true);
        
        // Set screw hole
        translate([0, hub_diameter/3, 0])
            rotate([90, 0, 0])
                cylinder(h = hub_diameter, d = shaft_diameter * 0.6, center = true);
    }
}

// Create smooth transition from hub to blade
module blade_root_transition(angle) {
    rotate([0, 0, angle]) {
        // Create a fillet between hub and blade root for structural integrity
        hull() {
            translate([blade_root_offset, 0, -hub_height/4])
                cylinder(h = 0.1, d = blade_chord_max * 0.6);
            
            translate([blade_root_offset - blade_chord_max * 0.1, 0, -hub_height/2])
                cylinder(h = 0.1, d = blade_chord_max * 0.3);
                
            translate([0, 0, -hub_height/2])
                cylinder(h = 0.1, d = hub_diameter);
        }
    }
}

// Complete propeller assembly
module propeller() {
    // Central hub
    hub();
    
    // Blades with transitions
    for (i = [0:num_blades-1]) {
        rotate([0, 0, i * 360 / num_blades]) {
            // Main blade
            blade();
            
            // Root transition for structural integrity
            blade_root_transition(0);
        }
    }
}

// Generate the propeller
propeller();
```