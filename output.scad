// Parametric Cup Implementation
// Author: Assistant
// Description: Modular cup design with ergonomic handle

// Quality Settings
$fn = 100;  // Smoothness factor for curved surfaces

// Main Cup Parameters
cup_height = 95;
cup_diameter = 75;
wall_thickness = 3.5;
base_thickness = 4;
taper_angle = 2;  // Degrees of taper for stability

// Handle Parameters
handle_thickness = 8;
handle_width = 15;
handle_height = 60;
handle_offset = cup_height * 0.3;  // Position from top
handle_curve_radius = 20;

// Calculated Values
cup_top_radius = cup_diameter / 2;
cup_bottom_radius = cup_top_radius - (cup_height * tan(taper_angle));
rim_fillet = 2;

module cup_body() {
    difference() {
        // Outer shell with taper
        cylinder(
            h = cup_height,
            r1 = cup_bottom_radius,
            r2 = cup_top_radius
        );
        
        // Interior hollow
        translate([0, 0, base_thickness])
            cylinder(
                h = cup_height,
                r1 = cup_bottom_radius - wall_thickness,
                r2 = cup_top_radius - wall_thickness
            );
            
        // Rim fillet
        translate([0, 0, cup_height - rim_fillet])
            rotate_extrude()
                translate([cup_top_radius - rim_fillet, 0, 0])
                    circle(r = rim_fillet);
    }
}

module handle() {
    translate([0, cup_top_radius - wall_thickness/2, cup_height - handle_offset]) {
        difference() {
            // Outer handle curve
            rotate([0, 0, 0])
                rotate_extrude(angle = 180)
                    translate([handle_curve_radius, 0, 0])
                        scale([1, 0.8])  // Slight vertical compression
                            circle(d = handle_thickness);
            
            // Inner cutout for grip
            rotate([0, 0, 0])
                rotate_extrude(angle = 180)
                    translate([handle_curve_radius, 0, 0])
                        scale([1, 0.8])
                            circle(d = handle_thickness * 0.6);
        }
    }
}

module base_reinforcement() {
    difference() {
        cylinder(
            h = base_thickness,
            r1 = cup_bottom_radius + 1,
            r2 = cup_bottom_radius
        );
        
        // Concave bottom for stability
        translate([0, 0, base_thickness/2])
            scale([1, 1, 0.3])
                sphere(r = cup_bottom_radius * 0.8);
    }
}

// Final Assembly
union() {
    cup_body();
    handle();
    base_reinforcement();
}