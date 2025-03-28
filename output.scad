// Parametric Drawer Model
// This model creates a simple drawer with a container housing

// Main parameters
drawer_width = 100;
drawer_depth = 120;
drawer_height = 60;
wall_thickness = 3;
clearance = 0.5;  // Gap between drawer and container

// Handle parameters
handle_diameter = 15;
handle_height = 10;
handle_position_y = drawer_depth * 0.5;

// Container parameters
container_extra_height = 10;  // How much taller the container is than the drawer

// Drawer module
module drawer() {
    difference() {
        // Outer drawer shell
        cube([drawer_width, drawer_depth, drawer_height]);
        
        // Inner cavity (slightly smaller than outer dimensions)
        translate([wall_thickness, wall_thickness, wall_thickness])
            cube([
                drawer_width - 2 * wall_thickness, 
                drawer_depth - 2 * wall_thickness, 
                drawer_height
            ]);
    }
    
    // Drawer front face (slightly thicker)
    difference() {
        translate([0, 0, 0])
            cube([drawer_width, wall_thickness * 2, drawer_height]);
        
        // Handle hole (if using a pull-through handle)
        // Uncomment if needed
        /*
        translate([drawer_width/2, 0, drawer_height/2])
            rotate([90, 0, 0])
                cylinder(h=wall_thickness*3, d=handle_diameter/2, center=true, $fn=30);
        */
    }
    
    // Drawer handle/knob
    translate([drawer_width/2, 0, drawer_height/2])
        rotate([90, 0, 0])
            cylinder(h=handle_height, d=handle_diameter, center=false, $fn=30);
}

// Container module
module container() {
    difference() {
        // Outer container shell
        cube([
            drawer_width + 2 * wall_thickness + 2 * clearance, 
            drawer_depth + wall_thickness + clearance, 
            drawer_height + container_extra_height + clearance
        ]);
        
        // Inner cavity (drawer space)
        translate([wall_thickness + clearance, 0, clearance])
            cube([
                drawer_width + clearance, 
                drawer_depth + clearance, 
                drawer_height + clearance
            ]);
        
        // Front opening
        translate([wall_thickness + clearance, -1, clearance])
            cube([
                drawer_width + clearance, 
                wall_thickness + 2, 
                drawer_height + clearance
            ]);
    }
}

// Assemble the model
// Container in position
color("SteelBlue", 0.8) container();

// Drawer pulled out slightly
color("Wheat") translate([wall_thickness + clearance, 30, clearance]) drawer();