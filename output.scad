// Propeller Model
// Units are in mm

// Parameters for customization
num_blades = 3;        // Number of propeller blades
blade_length = 50;     // Length of each blade from hub center
hub_diameter = 10;     // Diameter of the center hub
hub_height = 8;        // Height/thickness of the hub
blade_max_width = 15;  // Maximum width of the blade
blade_thickness = 2;   // Maximum thickness of the blade
blade_twist = 30;      // Twist angle from hub to tip (degrees)
shaft_hole = 3;        // Diameter of the center mounting hole
$fn = 80;              // Resolution for curved surfaces

// Module for the hub
module hub() {
    difference() {
        // Main hub cylinder
        cylinder(d=hub_diameter, h=hub_height, center=true);
        
        // Shaft hole through the center
        cylinder(d=shaft_hole, h=hub_height+1, center=true);
    }
}

// Module for a single blade
module blade() {
    // Create the blade using a series of transformed cross-sections
    for (i = [0:1:blade_length]) {
        // Calculate position along blade length (0.0 to 1.0)
        r_pos = i / blade_length;
        
        // Calculate width at this position (tapers toward tip)
        width = blade_max_width * (1 - pow(r_pos, 0.7));
        
        // Calculate thickness at this position (thinner toward tip)
        thick = blade_thickness * (1 - 0.7*pow(r_pos, 1.2));
        
        // Calculate twist angle at this position
        angle = r_pos * blade_twist;
        
        // Position along blade axis
        translate([i, 0, 0])
            // Apply twist around the blade axis
            rotate([0, 0, angle])
                // Create airfoil cross-section
                scale([thick, width, 1])
                    // Slightly rotate to create pitch
                    rotate([0, 90, 0])
                        // Use a stretched ellipse for the airfoil shape
                        resize([0, 1, 0.3])
                            circle(d=1);
    }
}

// Module for the complete propeller
module propeller() {
    // Add the hub
    hub();
    
    // Add the blades
    for (i = [0:num_blades-1]) {
        angle = i * 360 / num_blades;
        rotate([0, 0, angle])
            translate([hub_diameter/2-0.5, 0, 0])
                blade();
    }
}

// Render the propeller
propeller();