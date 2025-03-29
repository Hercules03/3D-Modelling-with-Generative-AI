// Computer Mouse Model
// A simple ergonomic computer mouse with primary and secondary buttons, scroll wheel, and curved body

// Main Parameters
mouse_length = 120;
mouse_width = 70;
mouse_height = 40;
mouse_curve = 0.6; // Controls the curvature of the mouse body (0-1)
roundness = 8;     // Controls the overall roundness of edges

// Button Parameters
primary_button_length = 40;
primary_button_width = 30;
secondary_button_length = 35;
secondary_button_width = 25;
button_height = 2;
button_gap = 1;

// Scroll Wheel Parameters
wheel_diameter = 12;
wheel_width = 8;
wheel_position = 35; // Distance from front

// Colors
body_color = "darkgray";
button_color = "lightgray";
wheel_color = "gray";

// Main mouse body module
module mouse_body() {
    difference() {
        // Base shape - rounded hull
        hull() {
            // Bottom corners
            translate([roundness, roundness, 0])
                cylinder(r=roundness, h=1, $fn=30);
            translate([mouse_length-roundness, roundness, 0])
                cylinder(r=roundness, h=1, $fn=30);
            translate([mouse_length-roundness, mouse_width-roundness, 0])
                cylinder(r=roundness, h=1, $fn=30);
            translate([roundness, mouse_width-roundness, 0])
                cylinder(r=roundness, h=1, $fn=30);
                
            // Top shape - curved profile
            scale([1, 1, 0.8]) translate([mouse_length/2, mouse_width/2, mouse_height])
                sphere(r=min(mouse_length, mouse_width)/2 * mouse_curve, $fn=50);
        }
        
        // Cut out the bottom to create a hollow shell
        translate([0, 0, -1])
            cube([mouse_length, mouse_width, 1]);
            
        // Button cutouts
        translate([button_gap, mouse_width/2 - primary_button_width/2 - button_gap/2, mouse_height/2])
            button_cutout(primary_button_length, primary_button_width);
            
        translate([button_gap, mouse_width/2 + button_gap/2, mouse_height/2])
            button_cutout(secondary_button_length, secondary_button_width);
            
        // Scroll wheel cutout
        translate([wheel_position, mouse_width/2, mouse_height/2 + 5])
            rotate([90, 0, 0])
                cylinder(d=wheel_diameter+2, h=wheel_width+2, center=true, $fn=30);
    }
}

// Button cutout shape
module button_cutout(length, width) {
    hull() {
        translate([0, 0, 0])
            cube([length, width, 1]);
        translate([length/5, width/5, mouse_height/2])
            cube([length*0.6, width*0.6, 1]);
    }
}

// Button module
module button(length, width) {
    hull() {
        cube([length, width, 1]);
        translate([length/5, width/5, button_height])
            cube([length*0.6, width*0.6, 0.01]);
    }
}

// Scroll wheel module
module scroll_wheel() {
    difference() {
        union() {
            // Main wheel
            cylinder(d=wheel_diameter, h=wheel_width, center=true, $fn=30);
            
            // Texture rings
            for (i = [-wheel_width/2+1.5 : 1.5 : wheel_width/2-1]) {
                translate([0, 0, i])
                    rotate_extrude($fn=30)
                        translate([wheel_diameter/2-0.5, 0, 0])
                            circle(r=0.5, $fn=10);
            }
        }
        
        // Axle hole
        cylinder(d=3, h=wheel_width+1, center=true, $fn=20);
    }
}

// Assemble the mouse
module assemble_mouse() {
    color(body_color) mouse_body();
    
    // Primary button
    color(button_color)
        translate([button_gap, mouse_width/2 - primary_button_width/2 - button_gap/2, mouse_height/2])
            button(primary_button_length, primary_button_width);
    
    // Secondary button
    color(button_color)
        translate([button_gap, mouse_width/2 + button_gap/2, mouse_height/2])
            button(secondary_button_length, secondary_button_width);
    
    // Scroll wheel
    color(wheel_color)
        translate([wheel_position, mouse_width/2, mouse_height/2 + 5])
            rotate([90, 0, 0])
                scroll_wheel();
}

// Create the mouse
assemble_mouse();