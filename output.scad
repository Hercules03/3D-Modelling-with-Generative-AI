// Parametric Bookshelf Model
// Adjustable dimensions for customization

/* Main Parameters */
// Shelf dimensions
width = 100;        // Width of the bookshelf
height = 180;       // Height of the bookshelf
depth = 30;         // Depth of the bookshelf
thickness = 2;      // Material thickness
num_shelves = 5;    // Number of shelves (including top and bottom)

// Calculated values
shelf_spacing = (height - thickness) / (num_shelves - 1);
shelf_height = thickness;

/* Modules */
// Create a single shelf component
module shelf() {
    cube([width, depth, shelf_height]);
}

// Create vertical supports
module vertical_support() {
    cube([thickness, depth, height]);
}

// Create a back panel
module back_panel() {
    translate([0, 0, 0])
    cube([width, thickness, height]);
}

// Create some sample books for the shelf
module book(w, d, h) {
    color([0.8, 0.4, 0.2])
    cube([w, d, h]);
}

/* Main Assembly */
// Left vertical support
translate([0, 0, 0])
vertical_support();

// Right vertical support
translate([width - thickness, 0, 0])
vertical_support();

// Back panel
translate([0, depth - thickness, 0])
back_panel();

// Shelves
for (i = [0:num_shelves-1]) {
    translate([0, 0, i * shelf_spacing])
    shelf();
}

// Sample books on shelves (for visualization)
// First shelf books
translate([5, 5, shelf_spacing + shelf_height]) book(8, 20, 25);
translate([15, 5, shelf_spacing + shelf_height]) book(10, 20, 22);
translate([30, 5, shelf_spacing + shelf_height]) book(12, 20, 24);
translate([45, 5, shelf_spacing + shelf_height]) book(9, 20, 26);
translate([60, 5, shelf_spacing + shelf_height]) book(15, 20, 20);

// Second shelf books
translate([10, 5, 2 * shelf_spacing + shelf_height]) book(20, 20, 18);
translate([35, 5, 2 * shelf_spacing + shelf_height]) book(18, 20, 20);
translate([60, 5, 2 * shelf_spacing + shelf_height]) book(25, 20, 15);

// Third shelf books
translate([5, 5, 3 * shelf_spacing + shelf_height]) book(30, 20, 22);
translate([40, 5, 3 * shelf_spacing + shelf_height]) book(25, 20, 24);
translate([70, 5, 3 * shelf_spacing + shelf_height]) book(22, 20, 20);