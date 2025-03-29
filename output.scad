// Parametric Lunch Box with Lid
// Author: OpenSCAD Expert
// Description: A simple lunch box with lid and compartments

/* Main Parameters */
// Box dimensions
box_length = 180;
box_width = 120;
box_height = 60;
wall_thickness = 2.5;
corner_radius = 10;

// Lid parameters
lid_height = 15;
lid_overlap = 5;
lid_tolerance = 0.5;

// Compartment parameters
use_compartments = true;
compartment_wall = 2;

// Hinge parameters
hinge_diameter = 5;
hinge_segments = 30;
hinge_gap = 0.5;

// Latch parameters
latch_width = 20;
latch_height = 10;
latch_thickness = 3;

// Resolution parameters
$fn = 40;

/* Main Modules */

// Create the lunch box
module lunch_box() {
    difference() {
        // Outer shell
        rounded_box(box_length, box_width, box_height, corner_radius);
        
        // Inner hollow
        translate([0, 0, wall_thickness])
            rounded_box(
                box_length - wall_thickness * 2, 
                box_width - wall_thickness * 2, 
                box_height, 
                corner_radius - wall_thickness
            );
        
        // Cut off top for hinge area
        translate([box_length/2 - 25, -box_width/2 - 1, box_height - 5])
            cube([50, box_width + 2, 10]);
    }
    
    // Add compartments if enabled
    if (use_compartments) {
        translate([0, 0, wall_thickness]) 
            compartments();
    }
    
    // Add hinge parts
    translate([box_length/2, -box_width/2, box_height - hinge_diameter/2])
        hinge_parts();
}

// Create the lid
module lid() {
    difference() {
        union() {
            // Main lid body
            rounded_box(
                box_length, 
                box_width, 
                lid_height, 
                corner_radius
            );
            
            // Inner lip for secure closure
            translate([0, 0, lid_height - lid_overlap])
                difference() {
                    rounded_box(
                        box_length - wall_thickness * 2 + lid_tolerance * 2, 
                        box_width - wall_thickness * 2 + lid_tolerance * 2, 
                        lid_overlap, 
                        corner_radius - wall_thickness
                    );
                    
                    translate([0, 0, -0.5])
                        rounded_box(
                            box_length - wall_thickness * 4, 
                            box_width - wall_thickness * 4, 
                            lid_overlap + 1, 
                            corner_radius - wall_thickness * 2
                        );
                }
        }
        
        // Hollow out the lid
        translate([0, 0, wall_thickness])
            rounded_box(
                box_length - wall_thickness * 2, 
                box_width - wall_thickness * 2, 
                lid_height, 
                corner_radius - wall_thickness
            );
        
        // Cut off bottom for hinge area
        translate([box_length/2 - 25, -box_width/2 - 1, -1])
            cube([50, box_width + 2, 10]);
    }
    
    // Add lid handle/knob
    translate([0, 0, lid_height])
        lid_handle();
    
    // Add hinge parts
    translate([box_length/2, -box_width/2, hinge_diameter/2])
        rotate([0, 0, 180])
            hinge_parts(is_lid = true);
    
    // Add latch
    translate([0, box_width/2, lid_height/2])
        lid_latch();
}

// Create a rounded box shape
module rounded_box(length, width, height, radius) {
    hull() {
        for (x = [-1, 1]) {
            for (y = [-1, 1]) {
                translate([
                    x * (length/2 - radius),
                    y * (width/2 - radius),
                    0
                ])
                cylinder(r = radius, h = height);
            }
        }
    }
}

// Create compartments inside the box
module compartments() {
    comp_length = (box_length - wall_thickness * 4 - compartment_wall) / 2;
    comp_width = box_width - wall_thickness * 2 - compartment_wall;
    comp_height = box_height - wall_thickness * 2;
    
    // Compartment divider
    translate([0, 0, comp_height/2])
        cube([compartment_wall, comp_width, comp_height], center = true);
    
    // Optional additional compartment divider
    translate([-comp_length/2 - compartment_wall/2, 0, comp_height/2])
        cube([comp_length, compartment_wall, comp_height], center = true);
}

// Create hinge parts
module hinge_parts(is_lid = false) {
    hinge_length = 50;
    hinge_segments = 5;
    segment_length = hinge_length / hinge_segments;
    
    for (i = [0:2:hinge_segments-1]) {
        translate([i * segment_length, 0, 0])
            rotate([0, 90, 0])
                cylinder(d = hinge_diameter, h = segment_length - hinge_gap);
    }
    
    if (is_lid) {
        for (i = [1:2:hinge_segments-1]) {
            translate([i * segment_length, 0, 0])
                rotate([0, 90, 0])
                    cylinder(d = hinge_diameter, h = segment_length - hinge_gap);
        }
    }
}

// Create lid handle/knob
module lid_handle() {
    handle_width = 40;
    handle_height = 10;
    
    translate([0, 0, 0]) {
        difference() {
            union() {
                // Base of handle
                scale([1, 0.8, 0.3])
                    sphere(r = handle_width / 2);
                
                // Flat bottom to attach to lid
                translate([0, 0, -handle_height/2])
                    cylinder(r = handle_width/3, h = handle_height/4);
            }
            
            // Cut off bottom half of sphere
            translate([0, 0, -handle_width/2])
                cube([handle_width * 2, handle_width * 2, handle_width], center = true);
        }
    }
}

// Create lid latch
module lid_latch() {
    translate([0, 0, 0])
        difference() {
            union() {
                // Latch base
                cube([latch_width, latch_thickness, latch_height], center = true);
                
                // Latch hook
                translate([0, latch_thickness/2 + 2, -latch_height/2 + 2])
                    rotate([90, 0, 0])
                        cylinder(r = 2, h = 4);
            }
            
            // Cutout for flexibility
            translate([0, 0, latch_height/4])
                cube([latch_width - 6, latch_thickness/2, latch_height/2], center = true);
        }
}

/* Main Assembly */
// Uncomment the desired part to render

// Render the box
lunch_box();

// Render the lid (translate it up to see separately)
translate([0, 0, box_height + 20])
    lid();

// Render the full assembly
// lunch_box();
// translate([0, 0, box_height])
//     rotate([0, 180, 0])
//         translate([0, 0, -lid_height])
//             lid();