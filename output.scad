```openscad
// Chair dimensions
seat_width = 500;
seat_depth = 450;
seat_height = 450;
back_height = 600;
leg_length = 400;
leg_width = 60;
back_width = 70;
back_thickness = 10;

// Leg parameters
leg_offset_x = seat_width / 2 - leg_width / 2;
leg_offset_y = seat_depth / 2 - leg_width / 2;

// Backrest parameters
back_offset_x = seat_width / 2 - back_width / 2;
back_offset_y = seat_depth / 2 - back_thickness / 2;

// Create chair
module chair() {
    // Legs
    translate([-leg_offset_x, -leg_offset_y, 0])
    cube([leg_width, leg_width, leg_length]);

    translate([leg_offset_x, -leg_offset_y, 0])
    cube([leg_width, leg_width, leg_length]);

    translate([-leg_offset_x, leg_offset_y, 0])
    cube([leg_width, leg_width, leg_length]);

    translate([leg_offset_x, leg_offset_y, 0])
    cube([leg_width, leg_width, leg_length]);

    translate([0, -leg_offset_y, 0])
    cube([leg_width, leg_width, leg_length]);

    // Seat
    translate([0, 0, leg_length])
    cube([seat_width, seat_depth, 20]);

    // Backrest
    translate([0, seat_depth - back_thickness, leg_length + 20])
    cube([back_width, back_thickness, back_height]);
}

// Render the chair
chair();
```