// Gate Parameters
gate_width = 200;            // Total width of the gate
gate_height = 300;           // Total height of the gate
post_thickness = 20;         // Thickness of the vertical posts
post_depth = 10;             // Depth of posts and frame
crossbar_thickness = 15;     // Thickness of the horizontal crossbar
crossbar_z = 180;            // Height at which the crossbar is positioned
hinge_radius = 3;            // Radius of the hinge cylinder
hinge_height = 25;           // Height (length) of the hinge cylinder
hinge_offset = 5;            // Offset from the left edge for the hinge
door_panel_depth = post_depth - 2; // Slightly inset door panel

// Module for a Vertical Post
module vertical_post() {
    // Creates a vertical post with base at [0,0,0]
    cube([post_thickness, post_depth, gate_height], center = false);
}

// Module for the Horizontal Crossbar
module crossbar() {
    // Creates the crossbar between the posts.
    // Its length spans the gap between the two posts.
    bar_length = gate_width - 2 * post_thickness;
    cube([bar_length, post_depth, crossbar_thickness], center = false);
}

// Module for the Door Panel
module door_panel() {
    // Creates an inset door panel occupying the space left by the posts and crossbar.
    // It is inset slightly along the depth for a neat fit.
    panel_width = gate_width - 2 * post_thickness;
    panel_height = crossbar_z; // Panel fills from base up to the crossbar height
    translate([post_thickness, 1, 0])
        cube([panel_width, door_panel_depth, panel_height], center = false);
}

// Module for a Hinge
module hinge() {
    // Creates a simple cylindrical hinge.
    // The hinge is centered along its height.
    cylinder(h = hinge_height, r = hinge_radius, center = true);
}

// Module to assemble the entire Gate
module gate() {
    // Left Vertical Post
    translate([0, 0, 0])
        vertical_post();
    
    // Right Vertical Post
    translate([gate_width - post_thickness, 0, 0])
        vertical_post();
        
    // Horizontal Crossbar positioned at crossbar_z
    translate([post_thickness, 0, crossbar_z])
        crossbar();
    
    // Door Panel (between posts, from base to crossbar)
    door_panel();
    
    // Hinge attached to the left post
    // Positioning the hinge on the left side, centered along Y and Z directions
    translate([hinge_offset, post_depth/2, gate_height/2])
        hinge();
}

// Render the Gate
gate();