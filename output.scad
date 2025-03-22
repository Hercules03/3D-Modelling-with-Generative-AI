// Mailbox Model in OpenSCAD
// Units are in mm

// Main dimensions
mailbox_length = 180;
mailbox_width = 80;
mailbox_base_height = 60;
mailbox_top_radius = 40;
wall_thickness = 3;

// Door dimensions
door_thickness = 2;
door_clearance = 1;

// Flag dimensions
flag_width = 30;
flag_height = 50;
flag_thickness = 2;
flag_pole_radius = 3;
flag_offset = 15;

// Post dimensions
post_width = 40;
post_height = 480;

// Main mailbox body module
module mailbox_body() {
    difference() {
        union() {
            // Base rectangular part
            cube([mailbox_length, mailbox_width, mailbox_base_height]);
            
            // Semi-cylindrical top
            translate([0, mailbox_width/2, mailbox_base_height])
                rotate([0, 90, 0])
                    cylinder(h=mailbox_length, r=mailbox_width/2);
        }
        
        // Hollow out the inside, leaving walls with thickness
        translate([wall_thickness, wall_thickness, wall_thickness])
            union() {
                cube([mailbox_length - 2*wall_thickness, 
                      mailbox_width - 2*wall_thickness, 
                      mailbox_base_height - wall_thickness]);
                
                translate([0, (mailbox_width - 2*wall_thickness)/2, mailbox_base_height - wall_thickness])
                    rotate([0, 90, 0])
                        cylinder(h=mailbox_length - 2*wall_thickness, 
                                r=(mailbox_width - 2*wall_thickness)/2);
            }
        
        // Door opening
        translate([-1, wall_thickness + door_clearance, wall_thickness + door_clearance])
            cube([wall_thickness + 2, 
                  mailbox_width - 2*wall_thickness - 2*door_clearance, 
                  mailbox_base_height - 2*wall_thickness - door_clearance]);
    }
}

// Door module
module door() {
    door_width = mailbox_width - 2*wall_thickness - 2*door_clearance;
    door_height = mailbox_base_height - 2*wall_thickness - door_clearance;
    
    difference() {
        union() {
            // Door panel
            cube([door_thickness, door_width, door_height]);
            
            // Door handle
            translate([door_thickness/2, door_width/2, door_height*0.7])
                rotate([0, 90, 0])
                    cylinder(h=10, r=5, center=true);
        }
        
        // Holes for hinges
        translate([-1, door_clearance, door_height*0.2])
            rotate([0, 90, 0])
                cylinder(h=door_thickness+2, r=2);
                
        translate([-1, door_clearance, door_height*0.8])
            rotate([0, 90, 0])
                cylinder(h=door_thickness+2, r=2);
    }
}

// Flag module
module flag() {
    // Flag pole
    cylinder(h=mailbox_base_height*0.6, r=flag_pole_radius);
    
    // Flag
    translate([0, flag_offset, mailbox_base_height*0.4])
        cube([flag_thickness, flag_width, flag_height]);
}

// Post module
module post() {
    translate([(mailbox_length - post_width)/2, (mailbox_width - post_width)/2, -post_height])
        cube([post_width, post_width, post_height]);
}

// Assemble the mailbox
module assembled_mailbox() {
    // Main body
    mailbox_body();
    
    // Door
    translate([wall_thickness, wall_thickness + door_clearance, wall_thickness + door_clearance])
        door();
    
    // Flag
    translate([mailbox_length*0.75, -flag_offset, mailbox_base_height*0.3])
        flag();
        
    // Post (optional - comment out if not needed)
    post();
}

// Render the mailbox
assembled_mailbox();