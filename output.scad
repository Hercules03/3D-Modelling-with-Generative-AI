// Propeller Design
// Customizable parameters
$fn = 100;  // Resolution for circular objects

// Main parameters
num_blades = 3;         // Number of propeller blades
hub_radius = 10;        // Radius of central hub
hub_height = 15;        // Height of central hub
shaft_radius = 3;       // Radius of shaft hole
blade_length = 50;      // Length of each blade
blade_width = 15;       // Maximum width of blade
blade_thickness = 3;    // Maximum thickness at blade root
twist_angle = 30;       // Twist angle from root to tip (degrees)
airfoil_camber = 0.1;   // Camber ratio for airfoil shape

// Module for creating a single propeller blade with twist and airfoil profile
module blade() {
    linear_extrude(height = blade_length, twist = -twist_angle, slices = 40, scale = 0.5) {
        union() {
            // Airfoil shape
            translate([-blade_width/4, 0, 0])
                scale([1, blade_thickness/blade_width, 1])
                    ellipse(blade_width/2, blade_width/2);
                
            // Add slight camber for improved aerodynamics
            translate([0, -blade_thickness * airfoil_camber, 0])
                scale([0.8, 0.2, 1])
                    circle(blade_width/2);
        }
    }
}

// Module for creating an ellipse
module ellipse(width, height) {
    scale([width, height, 1]) circle(1);
}

// Module for creating the central hub with shaft hole
module hub() {
    difference() {
        union() {
            // Main hub cylinder
            cylinder(r1 = hub_radius, r2 = hub_radius * 0.8, h = hub_height);
            
            // Hub base (for strength)
            cylinder(r = hub_radius * 1.2, h = hub_height * 0.2);
            
            // Hub top cap
            translate([0, 0, hub_height])
                cylinder(r1 = hub_radius * 0.8, r2 = hub_radius * 0.5, h = hub_height * 0.2);
        }
        
        // Shaft hole through center
        translate([0, 0, -1])
            cylinder(r = shaft_radius, h = hub_height + 2);
            
        // Setscrew hole
        translate([0, 0, hub_height/2])
            rotate([0, 90, 0])
                cylinder(r = shaft_radius/2, h = hub_radius + 1);
    }
}

// Assemble the propeller
module propeller() {
    // Central hub
    color("SlateGray") hub();
    
    // Blades
    color("LightSteelBlue")
    for (i = [0:num_blades-1]) {
        rotate([0, 0, i * 360 / num_blades])
            translate([hub_radius * 0.8, 0, hub_height * 0.6])
                rotate([0, 90, 0])
                    blade();
    }
}

// Create the propeller
propeller();