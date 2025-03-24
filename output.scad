// Simple Cabinet Model
// This model creates a parametric cabinet with doors, shelves, and handles

// Main cabinet parameters
cabinet_width = 100;     // Width of the cabinet
cabinet_height = 150;    // Height of the cabinet
cabinet_depth = 50;      // Depth of the cabinet
wall_thickness = 2;      // Thickness of the cabinet walls
num_shelves = 2;         // Number of shelves inside
door_gap = 1;            // Gap between doors and cabinet frame
handle_size = 5;         // Size of the door handles
has_doors = true;        // Whether to include doors

// Module for the main cabinet body
module cabinet_body() {
    difference() {
        // Outer shell
        cube([cabinet_width, cabinet_depth, cabinet_height]);
        
        // Inner hollow
        translate([wall_thickness, wall_thickness, wall_thickness])
            cube([
                cabinet_width - 2 * wall_thickness, 
                cabinet_depth - 2 * wall_thickness, 
                cabinet_height - wall_thickness
            ]);
    }
    
    // Add shelves
    for (i = [1:num_shelves]) {
        translate([wall_thickness, wall_thickness, i * cabinet_height / (num_shelves + 1)])
            cube([
                cabinet_width - 2 * wall_thickness, 
                cabinet_depth - 2 * wall_thickness, 
                wall_thickness
            ]);
    }
}

// Module for a door
module door(width, height) {
    cube([width, wall_thickness, height]);
    
    // Add handle
    translate([width / 2, wall_thickness, height / 2])
        rotate([90, 0, 0])
            cylinder(h = handle_size, r = handle_size / 2, center = false);
}

// Module for cabinet doors
module doors() {
    door_width = (cabinet_width - door_gap) / 2;
    door_height = cabinet_height - 2 * wall_thickness;
    
    // Left door
    translate([wall_thickness, cabinet_depth - 2 * wall_thickness, wall_thickness])
        door(door_width - door_gap, door_height);
    
    // Right door
    translate([cabinet_width / 2 + door_gap / 2, cabinet_depth - 2 * wall_thickness, wall_thickness])
        door(door_width - door_gap, door_height);
}

// Generate the complete cabinet
module cabinet() {
    cabinet_body();
    
    if (has_doors) {
        doors();
    }
}

// Render the cabinet
cabinet();