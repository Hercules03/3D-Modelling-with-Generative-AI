// Water Bottle with Cap
// Parameters for customization
bottle_height = 200;       // Height of the bottle (mm)
bottle_diameter = 70;      // Diameter of the bottle (mm)
wall_thickness = 2;        // Thickness of the bottle wall (mm)
neck_diameter = 30;        // Diameter of the bottle neck (mm)
neck_height = 20;          // Height of the bottle neck (mm)
cap_height = 15;           // Height of the cap (mm)
cap_diameter = neck_diameter + 4; // Diameter of the cap (mm)
thread_height = 10;        // Height of the thread section (mm)
thread_depth = 2;          // Depth of the thread (mm)
thread_rotations = 2;      // Number of thread rotations

// Main module for the water bottle
module water_bottle() {
    difference() {
        union() {
            // Main body of the bottle
            cylinder(h=bottle_height - neck_height, d=bottle_diameter, $fn=100);
            
            // Neck of the bottle
            translate([0, 0, bottle_height - neck_height])
                cylinder(h=neck_height, d1=bottle_diameter, d2=neck_diameter, $fn=100);
            
            // Thread for cap
            translate([0, 0, bottle_height - thread_height])
                thread(neck_diameter, thread_height, thread_depth, thread_rotations);
        }
        
        // Hollow out the bottle
        translate([0, 0, wall_thickness])
            cylinder(h=bottle_height - wall_thickness, d=bottle_diameter - 2*wall_thickness, $fn=100);
            
        // Hollow out the neck
        translate([0, 0, bottle_height - neck_height + wall_thickness])
            cylinder(h=neck_height, d1=bottle_diameter - 2*wall_thickness, 
                    d2=neck_diameter - 2*wall_thickness, $fn=100);
    }
}

// Module for the bottle cap
module bottle_cap() {
    difference() {
        // Cap outer shape
        union() {
            cylinder(h=cap_height, d=cap_diameter, $fn=100);
            
            // Add grip pattern on top of cap
            for(i = [0:30:359]) {
                rotate([0, 0, i])
                translate([cap_diameter/2 - 2, 0, cap_height - 1])
                    cylinder(h=2, d=4, $fn=20);
            }
        }
        
        // Hollow inside of cap
        translate([0, 0, wall_thickness])
            cylinder(h=cap_height, d=neck_diameter, $fn=100);
            
        // Thread cutout
        translate([0, 0, wall_thickness])
            thread(neck_diameter + thread_depth, thread_height, thread_depth, thread_rotations);
    }
}

// Module to create threads
module thread(diameter, height, depth, rotations) {
    pitch = height / rotations;
    
    for (i = [0:5:359*rotations]) {
        rotate([0, 0, i])
            translate([diameter/2, 0, i * pitch / 360])
                rotate([0, 90, 0])
                    cylinder(h=depth, d=pitch*0.8, $fn=20);
    }
}

// Render water bottle (comment/uncomment as needed)
water_bottle();

// Render cap (moved to display next to the bottle)
translate([bottle_diameter + 20, 0, 0])
    bottle_cap();