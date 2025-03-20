// Parametric Desk Design in OpenSCAD
// All measurements in millimeters

/* ===== PARAMETERS ===== */
// Desk dimensions
desk_width = 1200;      // 120cm wide
desk_depth = 600;       // 60cm deep
desk_height = 750;      // 75cm high
desk_thickness = 30;    // 3cm thick tabletop

// Leg dimensions
leg_width = 60;         // Square legs, 6cm wide
leg_height = desk_height - desk_thickness;
leg_inset = 30;         // Inset from the edges

// Cross-support dimensions
support_width = 40;
support_thickness = 20;
support_height_from_floor = 150;

// Cable management
cable_hole_diameter = 60;
cable_hole_offset = 150; // From the back edge

// Keyboard tray (optional)
include_keyboard_tray = true;
tray_width = 600;
tray_depth = 300;
tray_thickness = 20;
tray_height_from_floor = 550;

// Rounded corners
corner_radius = 20;     // Radius for tabletop corners

/* ===== MODULES ===== */
// Rounded rectangular tabletop
module rounded_tabletop() {
    minkowski() {
        cube([desk_width - 2*corner_radius, 
              desk_depth - 2*corner_radius, 
              desk_thickness - 1]);
        cylinder(r=corner_radius, h=1);
    }
}

// Desk leg
module leg() {
    cube([leg_width, leg_width, leg_height]);
}

// Cross-support beam
module cross_support(length) {
    cube([length, support_width, support_thickness]);
}

// Cable management hole
module cable_hole() {
    translate([0, 0, -1]) // Extend slightly below the surface
    cylinder(h=desk_thickness+2, d=cable_hole_diameter);
}

// Keyboard tray
module keyboard_tray() {
    translate([-tray_width/2, -tray_depth, 0])
    cube([tray_width, tray_depth, tray_thickness]);
}

/* ===== MAIN ASSEMBLY ===== */
module desk() {
    // Tabletop
    translate([corner_radius, corner_radius, leg_height])
    rounded_tabletop();
    
    // Legs - four corners
    // Front left leg
    translate([leg_inset, leg_inset, 0])
    leg();
    
    // Front right leg
    translate([desk_width - leg_inset - leg_width, leg_inset, 0])
    leg();
    
    // Back left leg
    translate([leg_inset, desk_depth - leg_inset - leg_width, 0])
    leg();
    
    // Back right leg
    translate([desk_width - leg_inset - leg_width, desk_depth - leg_inset - leg_width, 0])
    leg();
    
    // Cross-supports for stability
    // Front horizontal support
    translate([leg_inset + leg_width, leg_inset + leg_width/2 - support_width/2, support_height_from_floor])
    cross_support(desk_width - 2*leg_inset - 2*leg_width);
    
    // Back horizontal support
    translate([leg_inset + leg_width, desk_depth - leg_inset - leg_width/2 - support_width/2, support_height_from_floor])
    cross_support(desk_width - 2*leg_inset - 2*leg_width);
    
    // Left side support
    translate([leg_inset + leg_width/2 - support_width/2, leg_inset + leg_width, support_height_from_floor])
    rotate([0, 0, 90])
    cross_support(desk_depth - 2*leg_inset - 2*leg_width);
    
    // Right side support
    translate([desk_width - leg_inset - leg_width/2 - support_width/2, leg_inset + leg_width, support_height_from_floor])
    rotate([0, 0, 90])
    cross_support(desk_depth - 2*leg_inset - 2*leg_width);
    
    // Optional keyboard tray
    if (include_keyboard_tray) {
        translate([desk_width/2, tray_depth, tray_height_from_floor])
        keyboard_tray();
    }
    
    // Cable management holes
    difference() {
        // This is just a placeholder - we're using the difference() 
        // to cut the holes from the tabletop that was already placed
        cube([0.1, 0.1, 0.1]);
        
        // Center back cable hole
        translate([desk_width/2, cable_hole_offset, leg_height + desk_thickness/2])
        cable_hole();
        
        // Right cable hole
        translate([desk_width - cable_hole_offset, cable_hole_offset, leg_height + desk_thickness/2])
        cable_hole();
    }
}

/* ===== RENDER ===== */
// Render the complete desk
desk();