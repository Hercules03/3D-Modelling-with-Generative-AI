// Pirate Cutlass Sword Model
// This model creates a classic pirate cutlass with curved blade, cross-guard, and handle

// Parameters for customization
blade_length = 120;
blade_width = 15;
blade_thickness = 3;
blade_curve = 15;  // Amount of curvature in the blade
handle_length = 30;
handle_diameter = 10;
guard_width = 30;
guard_thickness = 4;
pommel_diameter = 12;
detail_resolution = 64;  // Smoothness of curved surfaces

// Main sword module
module pirate_sword() {
    union() {
        // Blade
        blade();
        
        // Cross-guard
        translate([0, 0, 0])
            cross_guard();
        
        // Handle
        translate([0, 0, -handle_length/2])
            handle();
        
        // Pommel
        translate([0, 0, -handle_length - 2])
            pommel();
    }
}

// Curved blade with beveled edge
module blade() {
    difference() {
        // Main blade shape - curved along the Y axis
        translate([0, blade_curve/2, blade_length/2])
        rotate([0, 0, 0])
        linear_extrude(height = blade_length, center = true, convexity = 10, twist = 0, scale = 0.6)
            translate([0, -blade_curve/blade_length*100, 0])
            resize([blade_width, blade_thickness])
            circle(d=10, $fn=detail_resolution);
        
        // Bevel for the cutting edge
        translate([0, blade_curve/2, blade_length/2])
        rotate([0, 0, 0])
        linear_extrude(height = blade_length*1.1, center = true, convexity = 10, twist = 0, scale = 0.5)
            translate([0, -blade_curve/blade_length*100, 0])
            resize([blade_width-1, blade_thickness-1])
            circle(d=10, $fn=detail_resolution);
    }
}

// Cross-guard with slight curve
module cross_guard() {
    difference() {
        union() {
            // Main guard bar
            translate([0, 0, 0])
            rotate([0, 90, 0])
            cylinder(h=guard_width, d=guard_thickness, center=true, $fn=detail_resolution);
            
            // Decorative center piece
            translate([0, 0, 0])
            sphere(d=guard_thickness*1.5, $fn=detail_resolution);
            
            // Curved guard ends
            for(i = [-1, 1]) {
                translate([i*guard_width/2, 0, 0])
                rotate([0, i*20, 0])
                rotate([0, 90, 0])
                cylinder(h=guard_width/4, d=guard_thickness, center=false, $fn=detail_resolution);
            }
        }
        
        // Hole for blade
        translate([0, 0, blade_length/6])
        cube([blade_width/2, blade_thickness*2, blade_length/3], center=true);
    }
}

// Handle with grip texture
module handle() {
    difference() {
        // Main handle cylinder
        cylinder(h=handle_length, d=handle_diameter, center=true, $fn=detail_resolution);
        
        // Grip texture
        for(i = [0:15:360]) {
            rotate([0, 0, i])
            translate([handle_diameter/2, 0, 0])
            rotate([90, 0, 0])
            cylinder(h=handle_diameter*2, d=1, center=true, $fn=8);
        }
    }
}

// Decorative pommel at end of handle
module pommel() {
    union() {
        sphere(d=pommel_diameter, $fn=detail_resolution);
        
        // Decorative cap
        translate([0, 0, pommel_diameter/3])
        rotate_extrude($fn=detail_resolution)
        polygon(points=[[0,0],[pommel_diameter/4,0],[pommel_diameter/6,pommel_diameter/4],[0,pommel_diameter/3]]);
    }
}

// Render the sword
pirate_sword();