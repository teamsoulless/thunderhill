var pi = 3.14159265358979;

/* Ellipsoid model constants (actual values here are for WGS84) */
var sm_a = 6378137.0;
var sm_b = 6356752.314;
var sm_EccSquared = 6.69437999013e-03;

var UTMScaleFactor = 0.9996;

/*
* DegToRad
*
* Converts degrees to radians.
*
*/
function DegToRad (deg)
{
    return (deg / 180.0 * pi)
}




/*
* RadToDeg
*
* Converts radians to degrees.
*
*/
function RadToDeg (rad)
{
    return (rad / pi * 180.0)
}




/*
* ArcLengthOfMeridian
*
* Computes the ellipsoidal distance from the equator to a point at a
* given latitude.
*
* Reference: Hoffmann-Wellenhof, B., Lichtenegger, H., and Collins, J.,
* GPS: Theory and Practice, 3rd ed.  New York: Springer-Verlag Wien, 1994.
*
* Inputs:
*     phi - Latitude of the point, in radians.
*
* Globals:
*     sm_a - Ellipsoid model major axis.
*     sm_b - Ellipsoid model minor axis.
*
* Returns:
*     The ellipsoidal distance of the point from the equator, in meters.
*
*/
function ArcLengthOfMeridian (phi)
{
    var alpha, beta, gamma, delta, epsilon, n;
    var result;

    /* Precalculate n */
    n = (sm_a - sm_b) / (sm_a + sm_b);

    /* Precalculate alpha */
    alpha = ((sm_a + sm_b) / 2.0)
       * (1.0 + (Math.pow (n, 2.0) / 4.0) + (Math.pow (n, 4.0) / 64.0));

    /* Precalculate beta */
    beta = (-3.0 * n / 2.0) + (9.0 * Math.pow (n, 3.0) / 16.0)
       + (-3.0 * Math.pow (n, 5.0) / 32.0);

    /* Precalculate gamma */
    gamma = (15.0 * Math.pow (n, 2.0) / 16.0)
        + (-15.0 * Math.pow (n, 4.0) / 32.0);

    /* Precalculate delta */
    delta = (-35.0 * Math.pow (n, 3.0) / 48.0)
        + (105.0 * Math.pow (n, 5.0) / 256.0);

    /* Precalculate epsilon */
    epsilon = (315.0 * Math.pow (n, 4.0) / 512.0);
    
    /* Now calculate the sum of the series and return */
    result = alpha
        * (phi + (beta * Math.sin (2.0 * phi))
        + (gamma * Math.sin (4.0 * phi))
        + (delta * Math.sin (6.0 * phi))
        + (epsilon * Math.sin (8.0 * phi)));

    return result;
}



/*
* UTMCentralMeridian
*
* Determines the central meridian for the given UTM zone.
*
* Inputs:
*     zone - An integer value designating the UTM zone, range [1,60].
*
* Returns:
*   The central meridian for the given UTM zone, in radians, or zero
*   if the UTM zone parameter is outside the range [1,60].
*   Range of the central meridian is the radian equivalent of [-177,+177].
*
*/
function UTMCentralMeridian (zone)
{
    var cmeridian;

    cmeridian = DegToRad (-183.0 + (zone * 6.0));

    return cmeridian;
}



/*
* FootpointLatitude
*
* Computes the footpoint latitude for use in converting transverse
* Mercator coordinates to ellipsoidal coordinates.
*
* Reference: Hoffmann-Wellenhof, B., Lichtenegger, H., and Collins, J.,
*   GPS: Theory and Practice, 3rd ed.  New York: Springer-Verlag Wien, 1994.
*
* Inputs:
*   y - The UTM northing coordinate, in meters.
*
* Returns:
*   The footpoint latitude, in radians.
*
*/
function FootpointLatitude (y)
{
    var y_, alpha_, beta_, gamma_, delta_, epsilon_, n;
    var result;
    
    /* Precalculate n (Eq. 10.18) */
    n = (sm_a - sm_b) / (sm_a + sm_b);

    /* Precalculate alpha_ (Eq. 10.22) */
    /* (Same as alpha in Eq. 10.17) */
    alpha_ = ((sm_a + sm_b) / 2.0)
        * (1 + (Math.pow (n, 2.0) / 4) + (Math.pow (n, 4.0) / 64));
    
    /* Precalculate y_ (Eq. 10.23) */
    y_ = y / alpha_;
    
    /* Precalculate beta_ (Eq. 10.22) */
    beta_ = (3.0 * n / 2.0) + (-27.0 * Math.pow (n, 3.0) / 32.0)
        + (269.0 * Math.pow (n, 5.0) / 512.0);
    
    /* Precalculate gamma_ (Eq. 10.22) */
    gamma_ = (21.0 * Math.pow (n, 2.0) / 16.0)
        + (-55.0 * Math.pow (n, 4.0) / 32.0);
    	
    /* Precalculate delta_ (Eq. 10.22) */
    delta_ = (151.0 * Math.pow (n, 3.0) / 96.0)
        + (-417.0 * Math.pow (n, 5.0) / 128.0);
    	
    /* Precalculate epsilon_ (Eq. 10.22) */
    epsilon_ = (1097.0 * Math.pow (n, 4.0) / 512.0);
    	
    /* Now calculate the sum of the series (Eq. 10.21) */
    result = y_ + (beta_ * Math.sin (2.0 * y_))
        + (gamma_ * Math.sin (4.0 * y_))
        + (delta_ * Math.sin (6.0 * y_))
        + (epsilon_ * Math.sin (8.0 * y_));
    
    return result;
}



/*
* MapLatLonToXY
*
* Converts a latitude/longitude pair to x and y coordinates in the
* Transverse Mercator projection.  Note that Transverse Mercator is not
* the same as UTM; a scale factor is required to convert between them.
*
* Reference: Hoffmann-Wellenhof, B., Lichtenegger, H., and Collins, J.,
* GPS: Theory and Practice, 3rd ed.  New York: Springer-Verlag Wien, 1994.
*
* Inputs:
*    phi - Latitude of the point, in radians.
*    lambda - Longitude of the point, in radians.
*    lambda0 - Longitude of the central meridian to be used, in radians.
*
* Outputs:
*    xy - A 2-element array containing the x and y coordinates
*         of the computed point.
*
* Returns:
*    The function does not return a value.
*
*/
function MapLatLonToXY (phi, lambda, lambda0, xy)
{
    var N, nu2, ep2, t, t2, l;
    var l3coef, l4coef, l5coef, l6coef, l7coef, l8coef;
    var tmp;

    /* Precalculate ep2 */
    ep2 = (Math.pow (sm_a, 2.0) - Math.pow (sm_b, 2.0)) / Math.pow (sm_b, 2.0);

    /* Precalculate nu2 */
    nu2 = ep2 * Math.pow (Math.cos (phi), 2.0);

    /* Precalculate N */
    N = Math.pow (sm_a, 2.0) / (sm_b * Math.sqrt (1 + nu2));

    /* Precalculate t */
    t = Math.tan (phi);
    t2 = t * t;
    tmp = (t2 * t2 * t2) - Math.pow (t, 6.0);

    /* Precalculate l */
    l = lambda - lambda0;

    /* Precalculate coefficients for l**n in the equations below
       so a normal human being can read the expressions for easting
       and northing
       -- l**1 and l**2 have coefficients of 1.0 */
    l3coef = 1.0 - t2 + nu2;

    l4coef = 5.0 - t2 + 9 * nu2 + 4.0 * (nu2 * nu2);

    l5coef = 5.0 - 18.0 * t2 + (t2 * t2) + 14.0 * nu2
        - 58.0 * t2 * nu2;

    l6coef = 61.0 - 58.0 * t2 + (t2 * t2) + 270.0 * nu2
        - 330.0 * t2 * nu2;

    l7coef = 61.0 - 479.0 * t2 + 179.0 * (t2 * t2) - (t2 * t2 * t2);

    l8coef = 1385.0 - 3111.0 * t2 + 543.0 * (t2 * t2) - (t2 * t2 * t2);

    /* Calculate easting (x) */
    xy[0] = N * Math.cos (phi) * l
        + (N / 6.0 * Math.pow (Math.cos (phi), 3.0) * l3coef * Math.pow (l, 3.0))
        + (N / 120.0 * Math.pow (Math.cos (phi), 5.0) * l5coef * Math.pow (l, 5.0))
        + (N / 5040.0 * Math.pow (Math.cos (phi), 7.0) * l7coef * Math.pow (l, 7.0));

    /* Calculate northing (y) */
    xy[1] = ArcLengthOfMeridian (phi)
        + (t / 2.0 * N * Math.pow (Math.cos (phi), 2.0) * Math.pow (l, 2.0))
        + (t / 24.0 * N * Math.pow (Math.cos (phi), 4.0) * l4coef * Math.pow (l, 4.0))
        + (t / 720.0 * N * Math.pow (Math.cos (phi), 6.0) * l6coef * Math.pow (l, 6.0))
        + (t / 40320.0 * N * Math.pow (Math.cos (phi), 8.0) * l8coef * Math.pow (l, 8.0));

    return;
}



/*
* MapXYToLatLon
*
* Converts x and y coordinates in the Transverse Mercator projection to
* a latitude/longitude pair.  Note that Transverse Mercator is not
* the same as UTM; a scale factor is required to convert between them.
*
* Reference: Hoffmann-Wellenhof, B., Lichtenegger, H., and Collins, J.,
*   GPS: Theory and Practice, 3rd ed.  New York: Springer-Verlag Wien, 1994.
*
* Inputs:
*   x - The easting of the point, in meters.
*   y - The northing of the point, in meters.
*   lambda0 - Longitude of the central meridian to be used, in radians.
*
* Outputs:
*   philambda - A 2-element containing the latitude and longitude
*               in radians.
*
* Returns:
*   The function does not return a value.
*
* Remarks:
*   The local variables Nf, nuf2, tf, and tf2 serve the same purpose as
*   N, nu2, t, and t2 in MapLatLonToXY, but they are computed with respect
*   to the footpoint latitude phif.
*
*   x1frac, x2frac, x2poly, x3poly, etc. are to enhance readability and
*   to optimize computations.
*
*/
function MapXYToLatLon (x, y, lambda0, philambda)
{
    var phif, Nf, Nfpow, nuf2, ep2, tf, tf2, tf4, cf;
    var x1frac, x2frac, x3frac, x4frac, x5frac, x6frac, x7frac, x8frac;
    var x2poly, x3poly, x4poly, x5poly, x6poly, x7poly, x8poly;

    /* Get the value of phif, the footpoint latitude. */
    f = FootpointLatitude (y);
   
    /* Precalculate ep2 */
    ep2 = (Math.pow (sm_a, 2.0) - Math.pow (sm_b, 2.0))
        / Math.pow (sm_b, 2.0);
        	
    /* Precalculate cos (phif) */
    cf = Math.cos (phif);
    	
    /* Precalculate nuf2 */
    nuf2 = ep2 * Math.pow (cf, 2.0);
    	
    /* Precalculate Nf and initialize Nfpow */
    Nf = Math.pow (sm_a, 2.0) / (sm_b * Math.sqrt (1 + nuf2));
    Nfpow = Nf;
    	
    /* Precalculate tf */
    tf = Math.tan (phif);
    tf2 = tf * tf;
    tf4 = tf2 * tf2;
    
    /* Precalculate fractional coefficients for x**n in the equations
       below to simplify the expressions for latitude and longitude. */
    x1frac = 1.0 / (Nfpow * cf);
    
    Nfpow *= Nf;   /* now equals Nf**2) */
    x2frac = tf / (2.0 * Nfpow);
    
    Nfpow *= Nf;   /* now equals Nf**3) */
    x3frac = 1.0 / (6.0 * Nfpow * cf);
    
    Nfpow *= Nf;   /* now equals Nf**4) */
    x4frac = tf / (24.0 * Nfpow);
    
    Nfpow *= Nf;   /* now equals Nf**5) */
    x5frac = 1.0 / (120.0 * Nfpow * cf);
    
    Nfpow *= Nf;   /* now equals Nf**6) */
    x6frac = tf / (720.0 * Nfpow);
    
    Nfpow *= Nf;   /* now equals Nf**7) */
    x7frac = 1.0 / (5040.0 * Nfpow * cf);
    
    Nfpow *= Nf;   /* now equals Nf**8) */
    x8frac = tf / (40320.0 * Nfpow);
    
    /* Precalculate polynomial coefficients for x**n.
       -- x**1 does not have a polynomial coefficient. */
    x2poly = -1.0 - nuf2;

    x3poly = -1.0 - 2 * tf2 - nuf2;

    x4poly = 5.0 + 3.0 * tf2 + 6.0 * nuf2 - 6.0 * tf2 * nuf2
    	- 3.0 * (nuf2 *nuf2) - 9.0 * tf2 * (nuf2 * nuf2);

    x5poly = 5.0 + 28.0 * tf2 + 24.0 * tf4 + 6.0 * nuf2 + 8.0 * tf2 * nuf2;
    
    x6poly = -61.0 - 90.0 * tf2 - 45.0 * tf4 - 107.0 * nuf2
    	+ 162.0 * tf2 * nuf2;
    
    x7poly = -61.0 - 662.0 * tf2 - 1320.0 * tf4 - 720.0 * (tf4 * tf2);
    
    x8poly = 1385.0 + 3633.0 * tf2 + 4095.0 * tf4 + 1575 * (tf4 * tf2);
    	
    /* Calculate latitude */
    philambda[0] = phif + x2frac * x2poly * (x * x)
    	+ x4frac * x4poly * Math.pow (x, 4.0)
    	+ x6frac * x6poly * Math.pow (x, 6.0)
    	+ x8frac * x8poly * Math.pow (x, 8.0);
    	
    /* Calculate longitude */
    philambda[1] = lambda0 + x1frac * x
    	+ x3frac * x3poly * Math.pow (x, 3.0)
    	+ x5frac * x5poly * Math.pow (x, 5.0)
    	+ x7frac * x7poly * Math.pow (x, 7.0);
    	
    return;
}




/*
* LatLonToUTMXY
*
* Converts a latitude/longitude pair to x and y coordinates in the
* Universal Transverse Mercator projection.
*
* Inputs:
*   lat - Latitude of the point, in radians.
*   lon - Longitude of the point, in radians.
*   zone - UTM zone to be used for calculating values for x and y.
*          If zone is less than 1 or greater than 60, the routine
*          will determine the appropriate zone from the value of lon.
*
* Outputs:
*   xy - A 2-element array where the UTM x and y values will be stored.
*
* Returns:
*   The UTM zone used for calculating the values of x and y.
*
*/
function LatLonToUTMXY (lat, lon, zone, xy)
{
    MapLatLonToXY (lat, lon, UTMCentralMeridian (zone), xy);

    /* Adjust easting and northing for UTM system. */
    xy[0] = xy[0] * UTMScaleFactor + 500000.0;
    xy[1] = xy[1] * UTMScaleFactor;
    if (xy[1] < 0.0)
        xy[1] = xy[1] + 10000000.0;

    return zone;
}



/*
* UTMXYToLatLon
*
* Converts x and y coordinates in the Universal Transverse Mercator
* projection to a latitude/longitude pair.
*
* Inputs:
*	x - The easting of the point, in meters.
*	y - The northing of the point, in meters.
*	zone - The UTM zone in which the point lies.
*	southhemi - True if the point is in the southern hemisphere;
*               false otherwise.
*
* Outputs:
*	latlon - A 2-element array containing the latitude and
*            longitude of the point, in radians.
*
* Returns:
*	The function does not return a value.
*
*/
function UTMXYToLatLon (x, y, zone, southhemi, latlon)
{
    var cmeridian;
    	
    x -= 500000.0;
    x /= UTMScaleFactor;
    	
    /* If in southern hemisphere, adjust y accordingly. */
    if (southhemi)
    y -= 10000000.0;
    		
    y /= UTMScaleFactor;
    
    cmeridian = UTMCentralMeridian (zone);
    MapXYToLatLon (x, y, cmeridian, latlon);
    	
    return;
}




/*
* btnToUTM_OnClick
*
* Called when the btnToUTM button is clicked.
*
*/
function latlon2xy(lon, lat)
{
    var xy = new Array(2);
    
    if ((lon < -180.0) || (180.0 <= lon)) {
        console.log ("The longitude you entered is out of range.  " +
               "Please enter a number in the range [-180, 180).");
        return false;
    }

    if ((lat < -90.0) || (90.0 < lat)) {
        console.log ("The latitude you entered is out of range.  " +
               "Please enter a number in the range [-90, 90].");
        return false;
    }

    // Compute the UTM zone.
    zone = Math.floor ((lon + 180.0) / 6) + 1;
    zone = LatLonToUTMXY (DegToRad (lat), DegToRad (lon), zone, xy);

    return xy;
}


/*
* btnToGeographic_OnClick
*
* Called when the btnToGeographic button is clicked.
*
*/
function xy2latlon (x, y, zone)
{                                  
    latlon = new Array(2);
    var x, y, zone, southhemi;

    if ((zone < 1) || (60 < zone)) {
        console.log ("The UTM zone you entered is out of range.  " +
               "Please enter a number in the range [1, 60].");
        return false;
    }

    southhemi = false;
    UTMXYToLatLon (x, y, zone, southhemi, latlon);

    return latlon;
}

var points = [
    [-122.3387906,39.5374557,0],
    [-122.3389462,39.5384693,0],
    [-122.3389864,39.5385582,0],
    [-122.3390454,39.5386182,0],
    [-122.3391178,39.5386658,0],
    [-122.3392171,39.5387072,0],
    [-122.3392949,39.5387444,0],
    [-122.3393539,39.538794,0],
    [-122.3393995,39.5388437,0],
    [-122.3394129,39.5388768,0],
    [-122.3396261,39.5391902,0],
    [-122.339673,39.5392677,0],
    [-122.339673,39.5392677,0],
    [-122.3396797,39.5393225,0],
    [-122.3396771,39.5393763,0],
    [-122.339665,39.5394187,0],
    [-122.3396301,39.5394756,0],
    [-122.3395993,39.5395066,0],
    [-122.3394477,39.5397218,0],
    [-122.3394035,39.5397745,0],
    [-122.3393512,39.5398097,0],
    [-122.3392868,39.5398438,0],
    [-122.3392144,39.5398697,0],
    [-122.3391487,39.539879,0],
    [-122.3390856,39.5398862,0],
    [-122.3388335,39.5398965,0],
    [-122.3386967,39.539911,0],
    [-122.3386028,39.5399338,0],
    [-122.3386028,39.5399338,0],
    [-122.3384875,39.5399824,0],
    [-122.3383883,39.5400372,0],
    [-122.3382957,39.5401127,0],
    [-122.3382407,39.5401717,0],
    [-122.3381911,39.5402461,0],
    [-122.3381576,39.5403206,0],
    [-122.3381375,39.5404033,0],
    [-122.3381442,39.5405067,0],
    [-122.3381589,39.5405936,0],
    [-122.3382019,39.5406743,0],
    [-122.3382448,39.5407394,0],
    [-122.3382998,39.5407912,0],
    [-122.3383467,39.5408305,0],
    [-122.3383936,39.5408656,0],
    [-122.3383936,39.5408656,0],
    [-122.3384768,39.5409122,0],
    [-122.338568,39.5409494,0],
    [-122.3386672,39.5409763,0],
    [-122.3387933,39.5409908,0],
    [-122.3389167,39.5409908,0],
    [-122.3390239,39.5409825,0],
    [-122.3391151,39.5409701,0],
    [-122.339209,39.5409287,0],
    [-122.3416337,39.5396732,0],
    [-122.341733,39.5396173,0],
    [-122.3418108,39.5395656,0],
    [-122.3418295,39.5394953,0],
    [-122.3418295,39.5394208,0],
    [-122.3418295,39.5393484,0],
    [-122.341851,39.5392595,0],
    [-122.3418885,39.5391891,0],
    [-122.3419476,39.5391043,0],
    [-122.34202,39.5390195,0],
    [-122.3421299,39.5389285,0],
    [-122.3422211,39.5388644,0],
    [-122.3423177,39.5387878,0],
    [-122.3423941,39.5387092,0],
    [-122.3424344,39.5386586,0],
    [-122.3424679,39.5386141,0],
    [-122.3425041,39.5385531,0],
    [-122.3425296,39.5385065,0],
    [-122.3425497,39.538462,0],
    [-122.3425658,39.5384227,0],
    [-122.3425846,39.5383679,0],
    [-122.3425953,39.5383069,0],
    [-122.3426047,39.5382604,0],
    [-122.3426127,39.5382169,0],
    [-122.3426194,39.538159,0],
    [-122.3426194,39.538099,0],
    [-122.3426168,39.5380504,0],
    [-122.3426087,39.5379935,0],
    [-122.342598,39.5379449,0],
    [-122.3425765,39.537887,0],
    [-122.3425591,39.5378343,0],
    [-122.3425591,39.5378343,0],
    [-122.3425336,39.5377763,0],
    [-122.3425055,39.5377381,0],
    [-122.3424558,39.5376998,0],
    [-122.3423874,39.5376584,0],
    [-122.3423043,39.5376284,0],
    [-122.3422386,39.5376129,0],
    [-122.3421447,39.5376057,0],
    [-122.3421447,39.5376057,0],
    [-122.3420669,39.5375933,0],
    [-122.3420052,39.5375757,0],
    [-122.3419288,39.5375478,0],
    [-122.3418604,39.5375116,0],
    [-122.3411067,39.5370792,0],
    [-122.3410571,39.5370451,0],
    [-122.3410088,39.5370068,0],
    [-122.3409592,39.5369603,0],
    [-122.3409592,39.5369603,0],
    [-122.3409149,39.5369096,0],
    [-122.3408854,39.536862,0],
    [-122.3408653,39.5368083,0],
    [-122.3408411,39.5367193,0],
    [-122.3408304,39.5366366,0],
    [-122.3408385,39.536558,0],
    [-122.3408639,39.5364473,0],
    [-122.3408639,39.5364473,0],
    [-122.3408827,39.536377,0],
    [-122.3408988,39.5363139,0],
    [-122.3409042,39.5362642,0],
    [-122.3408988,39.5361991,0],
    [-122.3408867,39.5361215,0],
    [-122.3408867,39.5361215,0],
    [-122.3408666,39.5360532,0],
    [-122.3407593,39.5357853,0],
    [-122.3407473,39.5357564,0],
    [-122.3407339,39.5357057,0],
    [-122.3407339,39.5357057,0],
    [-122.3407245,39.5356157,0],
    [-122.3407231,39.5355257,0],
    [-122.3407352,39.5354316,0],
    [-122.340758,39.535354,0],
    [-122.3407929,39.5352692,0],
    [-122.3408304,39.5351823,0],
    [-122.3411724,39.5345566,0],
    [-122.3411925,39.5345059,0],
    [-122.3412153,39.5344418,0],
    [-122.3412341,39.5343787,0],
    [-122.3412435,39.5343032,0],
    [-122.3412435,39.534238,0],
    [-122.3412381,39.5341646,0],
    [-122.341222,39.5340984,0],
    [-122.3411939,39.5340208,0],
    [-122.3411563,39.5339453,0],
    [-122.3411188,39.5338812,0],
    [-122.3410852,39.5338377,0],
    [-122.3409994,39.5337571,0],
    [-122.3409069,39.5336878,0],
    [-122.3409069,39.5336878,0],
    [-122.3408076,39.5336226,0],
    [-122.3406735,39.5335647,0],
    [-122.3404911,39.533515,0],
    [-122.3403329,39.5334902,0],
    [-122.3401746,39.5334674,0],
    [-122.3400351,39.5334364,0],
    [-122.3399091,39.533395,0],
    [-122.3398259,39.5333578,0],
    [-122.3397267,39.5333144,0],
    [-122.3396221,39.5332544,0],
    [-122.3395148,39.5331923,0],
    [-122.3394209,39.5331447,0],
    [-122.3393109,39.5331034,0],
    [-122.3391983,39.5330723,0],
    [-122.3390669,39.5330454,0],
    [-122.3389515,39.5330289,0],
    [-122.3388416,39.5330185,0],
    [-122.3387316,39.5330185,0],
    [-122.3387316,39.5330185,0],
    [-122.3386565,39.533031,0],
    [-122.3382702,39.5331116,0],
    [-122.3382045,39.5331509,0],
    [-122.3381737,39.5331996,0],
    [-122.338163,39.5332554,0],
    [-122.3381938,39.5333144,0],
    [-122.3382622,39.5333557,0],
    [-122.3383239,39.5333754,0],
    [-122.3387624,39.5335099,0],
    [-122.3388255,39.5335347,0],
    [-122.3389193,39.5335802,0],
    [-122.3389193,39.5335802,0],
    [-122.3390172,39.5336443,0],
    [-122.3390642,39.5336774,0],
    [-122.3399694,39.5344045,0],
    [-122.340011,39.5344449,0],
    [-122.3400486,39.5344883,0],
    [-122.3400915,39.5345411,0],
    [-122.340129,39.5346052,0],
    [-122.3401545,39.5346579,0],
    [-122.34018,39.5347366,0],
    [-122.3401974,39.5348038,0],
    [-122.3402081,39.5348721,0],
    [-122.3402068,39.5349238,0],
    [-122.3402068,39.5349724,0],
    [-122.3401974,39.5350241,0],
    [-122.340019,39.5358133,0],
    [-122.3400056,39.5358433,0],
    [-122.3399828,39.5358774,0],
    [-122.3399346,39.5359115,0],
    [-122.3398715,39.5359374,0],
    [-122.3398139,39.5359457,0],
    [-122.3397481,39.5359415,0],
    [-122.3396824,39.5359208,0],
    [-122.3396234,39.5358919,0],
    [-122.3395845,39.5358557,0],
    [-122.339555,39.5358122,0],
    [-122.3394276,39.5355123,0],
    [-122.3393646,39.5354285,0],
    [-122.3393096,39.5353758,0],
    [-122.3392533,39.5353344,0],
    [-122.3391755,39.5352847,0],
    [-122.3391058,39.5352547,0],
    [-122.3387302,39.5351399,0],
    [-122.3386136,39.5351399,0],
    [-122.338517,39.5351637,0],
    [-122.3384674,39.5352061,0],
    [-122.3384567,39.535293,0],
    [-122.3387906,39.5374557,0]    
];

console.log("UTM_path = [");
for (i=0; i < points.length; i++) {
    xy = latlon2xy(points[i][0], points[i][1]);
    console.log("    [%d, %d],", xy[0], xy[1]);
}
console.log("]");

