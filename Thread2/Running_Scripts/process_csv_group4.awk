#!/usr/bin/awk -f

BEGIN {
    FS = ",";  # Set field separator as comma for CSV
    OFS = ","; # Set output field separator as comma
}

NR > 1 { # Skip the first line (header)
    key = $1 OFS $2 OFS $3 OFS $4; # Group by the 1st and 2nd column
    count[key]++;
    sum5[key] += $5;
    sum8[key] += $8;
    sum9[key] += $9;
    sum10[key] += $10;
    sum11[key] += $11;
}

END {
    # Print the header for the output
    print "Chunk", "Overlap", "Dist", "TopN", "Time", "EM", "P", "R", "F1";
    
    # Iterate over the keys and print the means
    for (key in count) {
        mean5 = sum5[key] / count[key];
        mean8 = sum8[key] / count[key];
        mean9 = sum9[key] / count[key];
        mean10 = sum10[key] / count[key];
        mean11 = sum11[key] / count[key];
        print key, mean5, mean8, mean9, mean10, mean11;
    }
}
