Data
====

The GUI provides several windows for viewing and editing data arrays. This
chapter assumes you are familiar with *Tao* data nomenclature as discussed in
the Data chapter of the *Tao* manual.

Viewing Data
------------

The figures below shows the various windows that the GUI provides for viewing
data and making minor changes to data arrays.

.. figure:: /_static/gui/figures/view_data_1.png
   :align: center
   :alt: The GUI's data viewing windows.

   The Tao root window and the menu shortcut for viewing data.

.. figure:: /_static/gui/figures/view_data_2.png
   :align: center
   :alt: The GUI's data viewing windows.

   The d2_data_array window, which list the currently defined d2_data_arrays for
   each universe.
   This window also includes links to view and edit existing d1_data_arrays.

The d2_data_array window displays all data arrays for a given universe.
To view any existing d1_data_array, click on its "View" button.
This window also provides the ability to edit any existing data array in detail,
as well as functionality for writing existing data to a namelist.


.. figure:: /_static/gui/figures/view_data_4.png
   :align: center
   :alt: The GUI's data viewing windows.

   The d2_data_array window, which list the currently defined d2_data_arrays for
   each universe.


.. figure:: /_static/gui/figures/view_data_4.png
   :align: center
   :alt: The GUI's data viewing windows.

   The d1_data_array window (in this case for orbit.x), showing all of the
   datums in the array orbit.x.

The d1_data_array window allows the user to view an existing d1_data_array.
This window displays important properties of each datum in the array, such as
element name, meas, model, and design values, and weight, in a scrollable table.

To view a datum in detail, double click on its row in the d1_data_array window.
This will open the individual datum window for that datum, displaying all of its
properties and allowing some of them to be editted.


.. figure:: /_static/gui/figures/view_data_5.png
   :align: center
   :alt: The GUI's data viewing windows.

   The individual datum window (in this case for orbit.x[34]) displaying detailed datum properties and allowing the user to edit some of these properties.


.. figure:: /_static/gui/figures/view_data_3.png
   :align: center
   :alt: The GUI's data viewing windows.

   The bulk edit window (in this case for orbit.x) providing controls to quickly edit a few key properties for multiple datums in a d1_data_array.


The d1_data_array window also allows the user to edit a few key properties of
the datums in the array all at once using the bulk settings window.

This window is accessed by clicking on the "Bulk fill" button in the
d1_data_array window.

From here, the meas_value, good_user, and weight settings for the datums in the
array can be edited in bulk.

Changes may be applied to every datum in the array, or to only a specific range
of datums using the range specifier.

Once the desired settings have been specified, clicking the "Fill and apply"
button will edit the d1_data_array as necessary, and changes will be reflected
in the d1_data_array window.

Creating and Editing Data
-------------------------

*Tao* data structures can be defined via an initialization file or on-the-fly
via the GUI. For setting up a data initialization file, see the
*Tao Initialization* chapter in the *Tao* manual. To initialize data via the GUI,
open the **New D2 Data** window

The GUI also supports the creation of data arrays on the fly through the create
dat window.

This window can be accessed as shown in figure below.

.. figure:: /_static/gui/figures/create_d2_1.png
   :align: center
   :alt: Data creation window.

   The data creation window can be accessed from the root window's menubar.

.. figure:: /_static/gui/figures/create_d2_2.png
   :align: center
   :alt: Data creation window.

   The first pane of the data creation window.

In the first pane of the data creation window, the user can input the desired
settings for the new d2_data_array. The user can also select and existing
d2_data_array to clone.

This will copy the d2 properties of that array, as well as the d1 properties
and all of the datums for each d1 array.
Once this information has been input, the user can hit the "Next" button to go
to the d1_data_array pane.

The d1_array pane of the data creation window is where most of the data array's
properties are set.

.. figure:: /_static/gui/figures/create_d1_1.png
   :align: center
   :alt: The d1_array pane of the data creation window.

   Here, only one d1_array has been created (called my_d1), its default data
   type has been set to alpha.a, and the start and end indices have been set to
   1 and 12 respectively.

This pane is shown in the figure above.
The d1_array pane of the data creation window displays each d1_array in its own
tab. To add a tab, click on the "+" tab at the top of the window.
Tabs can also be removed by navigating to them and then clicking on their delete
button.
An existing tab can also be duplicated by clicking on the duplicate button right
under the delete button.
This may be useful if you want to define several d1_arrays with many of the same
properties, but want them each to have a different data type, for example.

The next section of the window holds the d1-level settings for the array.
Here, the d1 name, start index, and end index can be set, as well as the default
data_source, data_type, merit type, weight, and good user value for the d1_array.

The next section allows the users to set the ele_name, ele_start_name, and
ele_ref_name for the d1_array en-masse.

Clicking on these buttons will bring up the lattice browser window.

.. figure:: /_static/gui/figures/create_d1_2.png
   :align: center
   :alt: The d1_array pane of the data creation window.

   The lattice browser for the ele_names that will be used with my_d1.

This window is essentially identical to the main lattice window for the GUI,
with a few additions.

Towards the top right of the window, the user can specify which indices to read
the element names into.
Clicking "Apply Element Names" will then write the ele names that are currently
in the table sequentially into the d1_array's datums.
In the example shown, new_data_array.my_d1[1]|ele_name will be set to "Q00W\#1",
new_data_array.my_d1[2]|ele_name will be set to "Q01W", and so on.

If there are more elements in the table than there are datums to write to, the
table will be truncated and only the first elements in the table will be used.
If there are less elements in the table than there are datums to write to, the
elements in the table will be looped through so that each datum gets an element
name.

The bottom portion of the d1_array pane of the data creation window allows the
user to set the properties of the individual datums in the array.

Once a start and end index have been specified, the "Datum" drop down menu will
be populated with all of the datum indices.

Selecting an index will bring up the datum settings for that datum, as shown in
the figure below.

.. figure:: /_static/gui/figures/create_d1_3.png
   :align: center
   :alt: The d1_array pane of the data creation window.

   Here, the user has defined three d1_arrays: x, y, and z.
   The data type for new_data_array.x has been set to velocity.x, and the start
   and end indices have been set to 1 and 12.
   Here, the user is currently editing new_data_array.x[3], where the meas value
   has been set to 0.2 and the ref value has been set to 0.4.

Note that any settings that have a d1-level default value are automatically
filled in.
Once the user edits a property of a datum, that property will no longer be
auto-filled from the d1-level default settings, even if those default values are
subsequently edited.
If the user wants to explicitly fill a d1 setting to that d1_array's datums,
they may do so with the corresponding "Fill to datums" button.

Once all of the data settings have been adjusted as necessary, the user must
click the "Create" button to create the d2_array in Tao.  Doing so will close
the data creation window.


The data creation window can also be accessed from the d2_data window discussed
in Section ``Viewing Data``.


.. figure:: /_static/gui/figures/edit_data_1.png
   :align: center
   :alt: Editting an existing d2_data_array.

Clicking on the "Edit" button for any d2 array will load that array into the data
creation window, just as if the user had cloned that array from the d2 pane of
the data creation window.

.. figure:: /_static/gui/figures/edit_data_2.png
   :align: center
   :alt: Editting an existing d2_data_array.

   Editing an existing d2_data_array.

Note that any changes made in the data creation window will not take effect in
*Tao* until the user clicks the "Create" button.
For example, clicking the delete button for the orbit.x array would not actually
delete the array in Tao until the user clicks "Create".
