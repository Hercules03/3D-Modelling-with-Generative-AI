// Clip Design
// A simple clip with two arms that can be used to hold papers or other thin items
// The clip uses a spring mechanism for tension

// Parameters
clip_length = 50;         // Overall length of the clip
clip_width = 15;          // Width of the clip
clip_height = 7;          // Height of the clip body
wall_thickness = 2;       // Thickness of the walls
grip_length = 20;         // Length of the gripping part
spring_angle = 15;        // Angle for the spring tension (degrees)
grip_teeth_count = 5;     // Number of teeth on the gripping surface

// Main clip module
module clip() {
    difference() {
        union() {
            // Base body
            base();
            
            // Arms
            arms();
            
            // Grip teeth
            grip_teeth();
        }
        
        // Cutout to make the arms flexible
        translate([clip_length/4, 0, wall_thickness])
            cube([clip_length/2, clip_width-2*wall_thickness, clip_height]);
    }
}

// Base of the clip
module base() {
    hull() {
        // Rounded back
        translate([0, clip_width/2, clip_height/2])
            rotate([0, 90, 0])
                cylinder(h=wall_thickness, r=clip_height/2, $fn=30);
                
        // Main body
        translate([wall_thickness, 0, 0])
            cube([clip_length-wall_thickness, clip_width, wall_thickness]);
    }
}

// Arms of the clip
module arms() {
    // Bottom arm
    translate([wall_thickness, 0, 0])
        cube([clip_length-wall_thickness, wall_thickness, clip_height]);
    
    // Top arm
    translate([wall_thickness, clip_width-wall_thickness, 0])
        cube([clip_length-wall_thickness, wall_thickness, clip_height]);
    
    // Spring part (back)
    translate([0, 0, 0])
        cube([wall_thickness, clip_width, clip_height]);
    
    // Gripping ends
    translate([clip_length-grip_length, 0, 0]) {
        // Bottom grip
        translate([0, 0, 0])
            cube([grip_length, wall_thickness, clip_height]);
        
        // Top grip with angle for tension
        translate([0, clip_width-wall_thickness, 0])
            rotate([spring_angle, 0, 0])
                cube([grip_length, wall_thickness, clip_height+3]);
    }
}

// Teeth for better grip
module grip_teeth() {
    tooth_width = grip_length / grip_teeth_count;
    tooth_height = 1;
    
    for(i = [0:grip_teeth_count-1]) {
        // Bottom teeth
        translate([clip_length-grip_length+i*tooth_width, 0, clip_height])
            cube([tooth_width*0.7, wall_thickness, tooth_height]);
        
        // Top teeth (mirrored)
        translate([clip_length-grip_length+i*tooth_width, clip_width-wall_thickness, clip_height])
            rotate([spring_angle, 0, 0])
                cube([tooth_width*0.7, wall_thickness, tooth_height]);
    }
}

// Create the clip
clip();