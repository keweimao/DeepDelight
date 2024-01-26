#!/usr/bin/awk -f

BEGIN {
    FS = ",";  # Set field separator as comma for CSV
    OFS = ","; # Set output field separator as comma
}

NR > 1 { # Skip the first line (header)
    key = $1 OFS $2; # Group by the 1st and 2nd column
    count[key]++;
    sum3[key] += $3;
    sum6[key] += $6;
    sum7[key] += $7;
    sum8[key] += $8;
    sum9[key] += $9;
}

END {
    # Print the header for the output
    print "Group1", "Group2", "Mean3", "Mean6", "Mean7", "Mean8", "Mean9";
    
    # Iterate over the keys and print the means
    for (key in count) {
        mean3 = sum3[key] / count[key];
        mean6 = sum6[key] / count[key];
        mean7 = sum7[key] / count[key];
        mean8 = sum8[key] / count[key];
        mean9 = sum9[key] / count[key];
        print key, mean3, mean6, mean7, mean8, mean9;
    }
}
