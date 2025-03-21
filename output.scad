// Door dimensions (in mm)
door_height = 2032;  // 80 inches
door_width = 813;    // 32 inches
door_thickness = 44; // 1.75 inches

// Frame dimensions
frame_width = 89;    // 3.5 inches
frame_depth = 152;   // 6 inches
frame_gap = 3;       // Gap between door and frame

// Hinge dimensions
hinge_radius = 10;
hinge_height = 89;   // 3.5 inches
hinge_count = 3;

module door_panel() {
    cube([door_width, door_thickness, door_height]);
}

module door_frame() {
    difference() {
        // Outer frame
        cube([door_width + 2*frame_width, frame_depth, door_height + frame_width]);
        
        // Inner cutout
        translate([frame_width, 0, frame_width])
            cube([door_width + 2*frame_gap, frame_depth, door_height + frame_gap]);
    }
}

module hinge(z_pos) {
    translate([-hinge_radius/2, door_thickness/2, z_pos])
        rotate([0, 90, 0])
            cylinder(r=hinge_radius, h=hinge_radius*2);
}

module door_assembly() {
    // Door frame
    color("SaddleBrown")
        door_frame();
    
    // Door panel
    color("Sienna")
        translate([frame_width + frame_gap, frame_depth/4, frame_width])
            door_panel();
    
    // Hinges
    color("Silver")
        translate([frame_width, frame_depth/4, 0]) {
            hinge(door_height/6);
            hinge(door_height/2);
            hinge(5*door_height/6);
        }
}

// Create the complete door
door_assembly();