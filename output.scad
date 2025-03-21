// Parametric Coffee Mug Design
// Core dimensions
mug_height = 95;
mug_diameter = 65;
wall_thickness = 3.5;
bottom_thickness = 5;

// Handle parameters
handle_thickness = 8;
handle_width = 12;
handle_clearance = 10;
handle_vertical_span = 0.7; // Percentage of mug height

module mug_body() {
    difference() {
        // Outer shell
        cylinder(h=mug_height, d=mug_diameter, $fn=100);
        
        // Inner cavity
        translate([0, 0, bottom_thickness])
            cylinder(h=mug_height, 
                    d=mug_diameter - (wall_thickness * 2), 
                    $fn=100);
    }
}

module handle() {
    // Handle attachment points
    handle_height = mug_height * handle_vertical_span;
    handle_offset = mug_diameter/2 + handle_clearance;
    
    // Create smooth handle using hull() between spheres
    translate([mug_diameter/2, 0, mug_height * 0.8])
    hull() {
        // Top connection point
        sphere(d=handle_thickness, $fn=30);
        
        // Outer curve points
        translate([handle_offset-mug_diameter/2, 0, -handle_height/3])
            sphere(d=handle_thickness, $fn=30);
        
        translate([handle_offset-mug_diameter/2, 0, -handle_height*2/3])
            sphere(d=handle_thickness, $fn=30);
            
        // Bottom connection point
        translate([0, 0, -handle_height])
            sphere(d=handle_thickness, $fn=30);
    }
}

// Final assembly
module complete_mug() {
    union() {
        mug_body();
        handle();
    }
}

// Render mug
complete_mug();