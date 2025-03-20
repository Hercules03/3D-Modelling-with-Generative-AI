// Book Shelf Implementation in OpenSCAD

// Parameters for the bookshelf
bookShelfWidth = 100;       // Width of the bookshelf (X-axis)
bookShelfHeight = 200;      // Height of the bookshelf (Y-axis)
bookShelfDepth = 30;        // Depth of the bookshelf (Z-axis)
shelfThickness = 2;         // Thickness of each shelf
sideThickness = 3;          // Thickness of the side panels
backThickness = 1.5;        // Thickness of the back panel
numShelves = 5;             // Number of shelves
wallSpacing = (bookShelfHeight - shelfThickness) / (numShelves + 1); // Spacing between shelves

// Module for side panels
module sidePanel() {
    cube([sideThickness, bookShelfHeight, bookShelfDepth], center = false);
}

// Module for a single shelf
module shelf() {
    cube([bookShelfWidth - 2 * sideThickness, shelfThickness, bookShelfDepth], center = false);
}

// Module for the back panel
module backPanel() {
    cube([bookShelfWidth, bookShelfHeight, backThickness], center = false);
}

// Assembly of the bookshelf
module bookShelfAssembly() {
    // Create side panels
    translate([0, 0, 0])
        sidePanel();
    translate([bookShelfWidth - sideThickness, 0, 0])
        sidePanel();
    
    // Create shelves
    for (i = [1 : numShelves]) {
        translate([sideThickness, wallSpacing * i, 0])
            shelf();
    }
    
    // Create back panel
    translate([0, 0, bookShelfDepth - backThickness])
        backPanel();
}

// Render the bookshelf
bookShelfAssembly();