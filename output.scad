// Drone Propeller
// Parametric design for a standard quadcopter propeller
// Includes hub, airfoil blades with proper pitch, and mounting hole

// Resolution for circular objects
$fn = 100;

// Main parameters
num_blades = 2;         // Number of propeller blades
blade_length = 65;      // Length of each blade from hub center
hub_radius = 10;        // Radius of central hub
hub_height = 8;         // Height of central hub
shaft_radius = 3.2;     // Radius of motor shaft hole
shaft_flat = true;      // Whether shaft has a flat side for mounting
blade_width = 15;       // Maximum width of blade
blade_thickness = 2.5;  // Maximum thickness at blade root
twist_angle = 35;       // Twist angle from root to tip (degrees)
airfoil_camber = 0.08;  // Camber ratio for airfoil shape
rotation_dir = 1;       // 1 for CW, -1 for CCW rotation

// Module for creating a single propeller blade with twist and airfoil profile
module blade() {
    // Create blade with twist and taper
    hull() {
        for (i = [0:10]) {
            pos = i/10;
            // Calculate position along blade
            x_pos = pos * blade_length;
            
            // Calculate blade width and thickness based on position
            local_width = blade_width * (1 - 0.6*pos);
            local_thickness = blade_thickness * (1 - 0.5*pos);
            
            // Calculate twist angle
            local_twist = twist_angle * pos * rotation_dir;
            
            translate([x_pos, 0, 0])
            rotate([0, 0, local_twist])
            scale([1, local_thickness/local_width, 1])
            translate([0, 0, -local_thickness/2])
            union() {
                // Main airfoil shape
                scale([local_width/2, 1, local_thickness])
                rotate([0, 90, 0])
                cylinder(h=0.01, r1=1, r2=1);
                
                // Add slight camber for improved aerodynamics
                translate([0, local_width * airfoil_camber, 0])
                scale([local_width/2, 1, local_thickness * 0.8])
                rotate([0, 90, 0])
                cylinder(h=0.01, r1=0.8, r2=0.8);
            }
        }
    }
}

// Module for creating the central hub with shaft hole
module hub() {
    difference() {
        union() {
            // Main hub cylinder
            cylinder(r1 = hub_radius, r2 = hub_radius * 0.9, h = hub_height);
            
            // Hub base (for strength)
            cylinder(r = hub_radius * 1.1, h = hub_height * 0.3);
            
            // Top reinforcement
            translate([0, 0, hub_height - hub_height * 0.2])
                cylinder(r1 = hub_radius * 0.9, r2 = hub_radius * 0.7, h = hub_height * 0.2);
        }
        
        // Shaft hole through center
        translate([0, 0, -1]) {
            cylinder(r = shaft_radius, h = hub_height + 2);
            
            // Add flat side for motor shaft if needed
            if (shaft_flat) {
                translate([shaft_radius * 0.6, 0, 0])
                    cube([shaft_radius * 0.8, shaft_radius * 2, hub_height + 2], center=true);
            }
        }
        
        // Mounting screw holes (optional)
        for (i = [0:3]) {
            rotate([0, 0, i * 90])
            translate([hub_radius * 0.6, 0, hub_height/2])
            rotate([0, 90, 0])
            cylinder(r = 1.5, h = hub_radius);
        }
    }
}

// Blade with fillet connection to hub
module blade_with_fillet() {
    union() {
        blade();
        
        // Create fillet at blade root
        translate([0, 0, -blade_thickness/2])
        hull() {
            translate([0, 0, 0])
            scale([hub_radius * 0.4, blade_width * 0.4, blade_thickness])
            sphere(r = 1);
            
            translate([blade_length * 0.15, 0, 0])
            scale([hub_radius * 0.2, blade_width * 0.6, blade_thickness])
            sphere(r = 1);
        }
    }
}

// Assemble the propeller
module propeller() {
    // Central hub
    hub();
    
    // Blades with even spacing
    for (i = [0:num_blades-1]) {
        rotate([0, 0, i * (360 / num_blades)])
        translate([hub_radius, 0, hub_height/2])
        blade_with_fillet();
    }
}

// Create the propeller
propeller();