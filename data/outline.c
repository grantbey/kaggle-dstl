/*
Modifioitu pygame.mask.Mask.outline
Input: data, width, height, every
Source: http://stackoverflow.com/a/14223101/5441199
*/

PyObject *plist, *value;
int x, y, e, firstx, firsty, secx, secy, currx, curry, nextx, nexty, n;
int a[14], b[14];
a[0] = a[1] = a[7] = a[8] = a[9] = b[1] = b[2] = b[3] = b[9] = b[10] = b[11]= 1;
a[2] = a[6] = a[10] = b[4] = b[0] = b[12] = b[8] = 0;
a[3] = a[4] = a[5] = a[11] = a[12] = a[13] = b[5] = b[6] = b[7] = b[13] = -1;

plist = NULL;
plist = PyList_New (0);
/*if (!plist) En ymmärrä mihin tätä tarvii
    return NULL;*/

every = 1;
n = firstx = firsty = secx = x = 0;

/*if(!PyArg_ParseTuple(args, "|i", &every)) {
    return NULL;
}

 by copying to a new, larger mask, we avoid having to check if we are at
   a border pixel every time.  
bitmask_draw(m, c, 1, 1); */

e = every;

/* find the first set pixel in the mask */
for (y = 1; y < height-1; y++) {
    for (x = 1; x < width-1; x++) {
        if (data(x, y)) {
             firstx = x;
             firsty = y;
             value = Py_BuildValue("(ii)", x-1, y-1);
             PyList_Append(plist, value);
             Py_DECREF(value);
             break;
        }
    }
    if (data(x, y))
        break;
}



/* covers the mask having zero pixels or only the final pixel
Pikseleitä on ainakin kymmenen
if ((x == width-1) && (y == height-1)) {
    return plist;
}        */

/* check just the first pixel for neighbors */
for (n = 0;n < 8;n++) {
    if (data(x+a[n], y+b[n])) {
        currx = secx = x+a[n];
        curry = secy = y+b[n];
        e--;
        if (!e) {
            e = every;
            value = Py_BuildValue("(ii)", secx-1, secy-1);
            PyList_Append(plist, value);
            Py_DECREF(value);
        }
        break;
    }
}       

/* if there are no neighbors, return
Pikseleitä on ainakin kymmenen
if (!secx) {
    return plist;
}*/

/* the outline tracing loop */
for (;;) {
    /* look around the pixel, it has to have a neighbor */
    for (n = (n + 6) & 7;;n++) {
        if (data(currx+a[n], curry+b[n])) {
            nextx = currx+a[n];
            nexty = curry+b[n];
            e--;
            if (!e) {
                e = every;
                if ((curry == firsty && currx == firstx) && (secx == nextx && secy == nexty)) {
                    break;
                }
                value = Py_BuildValue("(ii)", nextx-1, nexty-1);
                PyList_Append(plist, value);
                Py_DECREF(value);
            }
            break;
        }
    }
    /* if we are back at the first pixel, and the next one will be the
       second one we visited, we are done */
    if ((curry == firsty && currx == firstx) && (secx == nextx && secy == nexty)) {
        break;
    }

    curry = nexty;
    currx = nextx;
}

return_val = plist;