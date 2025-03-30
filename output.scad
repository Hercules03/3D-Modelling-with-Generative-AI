// Drone Propeller Generator
// Customizable parameters for creating a drone propeller

/* MAIN PARAMETERS */
// Basic propeller dimensions
num_blades = 2;          // Number of propeller blades
blade_length = 50;       // Length of each blade from hub center (mm)
hub_diameter = 12;       // Diameter of the central hub (mm)
hub_height = 8;          // Height of the central hub (mm)
shaft_diameter = 5;      // Diameter of the shaft hole (mm)

// Blade shape parameters
blade_width_root = 12;   // Width of blade at the root (mm)
blade_width_tip = 6;     // Width of blade at the tip (mm)
blade_thickness = 2;     // Maximum blade thickness (mm)
pitch_angle = 25;        // Base pitch angle (degrees)
twist_angle = 15;        // Additional twist from root to tip (degrees)

// Resolution parameters
$fn = 60;                // Default resolution for curved surfaces
blade_segments = 20;     // Number of segments along blade length

/* MODULES */

// Generate an airfoil cross-section
module airfoil(chord, thickness, camber=0.05) {
    // NACA-inspired airfoil profile
    points = [
        // Upper surface points (leading to trailing edge)
        [0, 0],
        [0.025*chord, 0.3*thickness],
        [0.05*chord, 0.5*thickness],
        [0.1*chord, 0.8*thickness],
        [0.2*chord, thickness],
        [0.3*chord, 0.95*thickness],
        [0.5*chord, 0.8*thickness],
        [0.7*chord, 0.5*thickness],
        [0.9*chord, 0.2*thickness],
        [chord, 0],
        // Lower surface points (trailing to leading edge)
        [0.9*chord, -0.15*thickness],
        [0.7*chord, -0.3*thickness],
        [0.5*chord, -0.4*thickness],
        [0.3*chord, -0.5*thickness],
        [0.2*chord, -0.45*thickness],
        [0.1*chord, -0.3*thickness],
        [0.05*chord, -0.2*thickness],
        [0.025*chord, -0.1*thickness],
        [0, 0]
    ];
    
    // Apply camber
    camber_points = [for (p = points) 
        let (x = p[0], 
             y = p[1],
             camber_y = camber * sin(180 * x/chord))
        [x, y + camber_y]
    ];
    
    polygon(camber_points);
}

// Create a single blade
module blade() {
    for (i = [0:blade_segments-1]) {
        t1 = i / blade_segments;
        t2 = (i + 1) / blade_segments;
        
        z1 = t1 * blade_length;
        z2 = t2 * blade_length;
        
        // Calculate chord width (linear taper)
        chord1 = blade_width_root + t1 * (blade_width_tip - blade_width_root);
        chord2 = blade_width_root + t2 * (blade_width_tip - blade_width_root);
        
        // Calculate thickness (thinner toward tip)
        thickness1 = blade_thickness * (1 - 0.5 * t1);
        thickness2 = blade_thickness * (1 - 0.5 * t2);
        
        // Calculate twist angle (varies along length)
        angle1 = pitch_angle + t1 * twist_angle;
        angle2 = pitch_angle + t2 * twist_angle;
        
        // Create blade segment
        hull() {
            translate([0, 0, z1])
                rotate([0, 0, angle1])
                    linear_extrude(height=0.01)
                        airfoil(chord1, thickness1);
                        
            translate([0, 0, z2])
                rotate([0, 0, angle2])
                    linear_extrude(height=0.01)
                        airfoil(chord2, thickness2);
        }
    }
}

// Create hub with mounting hole
module hub() {
    difference() {
        union() {
            // Main hub cylinder
            cylinder(h=hub_height, d=hub_diameter, center=true);
            
            // Top hub cap (optional)
            translate([0, 0, hub_height/2])
                cylinder(h=hub_height/4, d1=hub_diameter, d2=hub_diameter*0.8, center=true);
                
            // Bottom hub cap (optional)
            translate([0, 0, -hub_height/2])
                cylinder(h=hub_height/4, d1=hub_diameter*0.8, d2=hub_diameter, center=true);
        }
        
        // Shaft hole
        cylinder(h=hub_height*1.2, d=shaft_diameter, center=true);
        
        // Set screw hole (optional)
        translate([0, hub_diameter/2, 0])
            rotate([90, 0, 0])
                cylinder(h=hub_diameter, d=2, center=true);
    }
}

// Create the complete propeller
module propeller() {
    // Add the hub
    color("SlateGray") hub();
    
    // Add the blades
    color("LightSteelBlue")
    for (i = [0:num_blades-1]) {
        rotate([0, 0, i * 360 / num_blades])
            translate([0, 0, 0])
                rotate([90, 0, 0])
                    blade();
    }
}

// Generate the propeller
propeller();