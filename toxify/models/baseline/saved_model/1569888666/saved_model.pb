│╓

└8Щ8
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	АР
ю
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
Ы
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

ControlTrigger
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

B
Equal
x"T
y"T
z
"
Ttype:
2	
Р
)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
Ц
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
:
Less
x"T
y"T
z
"
Ttype:
2	
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
$

LogicalAnd
x

y

z
Р
!
LoopCond	
input


output

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	Р
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
;
Minimum
x"T
y"T
z"T"
Ttype:

2	Р
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
A

StackPopV2

handle
elem"	elem_type"
	elem_typetypeИ
X
StackPushV2

handle	
elem"T
output"T"	
Ttype"
swap_memorybool( И
S
StackV2
max_size

handle"
	elem_typetype"

stack_namestring И
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:И
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestringИ
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetypeИ
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
TtypeИ
9
TensorArraySizeV3

handle
flow_in
sizeИ
▐
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring И
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
TtypeИ
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
┴
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments"Tnumsegments
output"T" 
Ttype:
2	"
Tindicestype:
2	" 
Tnumsegmentstype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.8.02v1.8.0-0-g93bc2e2072Ам	
И
PlaceholderPlaceholder*
dtype0*4
_output_shapes"
 :                  *)
shape :                  
p
Placeholder_1Placeholder*
shape:         *
dtype0*'
_output_shapes
:         
J
rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Q
rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
Q
rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
f
	rnn/rangeRangernn/range/startrnn/Rankrnn/range/delta*
_output_shapes
:*

Tidx0
d
rnn/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
Q
rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Б

rnn/concatConcatV2rnn/concat/values_0	rnn/rangernn/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0

rnn/transpose	TransposePlaceholder
rnn/concat*
T0*4
_output_shapes"
 :                  *
Tperm0
V
	rnn/ShapeShapernn/transpose*
T0*
out_type0*
_output_shapes
:
a
rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
c
rnn/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
c
rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Н
rnn/strided_sliceStridedSlice	rnn/Shapernn/strided_slice/stackrnn/strided_slice/stack_1rnn/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
e
#rnn/GRUCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ц
rnn/GRUCellZeroState/ExpandDims
ExpandDimsrnn/strided_slice#rnn/GRUCellZeroState/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
e
rnn/GRUCellZeroState/ConstConst*
valueB:О*
dtype0*
_output_shapes
:
b
 rnn/GRUCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
└
rnn/GRUCellZeroState/concatConcatV2rnn/GRUCellZeroState/ExpandDimsrnn/GRUCellZeroState/Const rnn/GRUCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
e
 rnn/GRUCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ж
rnn/GRUCellZeroState/zerosFillrnn/GRUCellZeroState/concat rnn/GRUCellZeroState/zeros/Const*
T0*

index_type0*(
_output_shapes
:         О
g
%rnn/GRUCellZeroState/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 
Ъ
!rnn/GRUCellZeroState/ExpandDims_1
ExpandDimsrnn/strided_slice%rnn/GRUCellZeroState/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
g
rnn/GRUCellZeroState/Const_1Const*
valueB:О*
dtype0*
_output_shapes
:
X
rnn/Shape_1Shapernn/transpose*
T0*
out_type0*
_output_shapes
:
c
rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
e
rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
e
rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ч
rnn/strided_slice_1StridedSlicernn/Shape_1rnn/strided_slice_1/stackrnn/strided_slice_1/stack_1rnn/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
X
rnn/Shape_2Shapernn/transpose*
T0*
out_type0*
_output_shapes
:
c
rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
e
rnn/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
e
rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ч
rnn/strided_slice_2StridedSlicernn/Shape_2rnn/strided_slice_2/stackrnn/strided_slice_2/stack_1rnn/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
T
rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
v
rnn/ExpandDims
ExpandDimsrnn/strided_slice_2rnn/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
T
	rnn/ConstConst*
valueB:О*
dtype0*
_output_shapes
:
S
rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
А
rnn/concat_1ConcatV2rnn/ExpandDims	rnn/Constrnn/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
T
rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
u
	rnn/zerosFillrnn/concat_1rnn/zeros/Const*
T0*

index_type0*(
_output_shapes
:         О
J
rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
Д
rnn/TensorArrayTensorArrayV3rnn/strided_slice_1*%
element_shape:         О*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*/
tensor_array_namernn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: 
Д
rnn/TensorArray_1TensorArrayV3rnn/strided_slice_1*
identical_element_shapes(*.
tensor_array_namernn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *$
element_shape:         *
clear_after_read(*
dynamic_size( 
i
rnn/TensorArrayUnstack/ShapeShapernn/transpose*
T0*
out_type0*
_output_shapes
:
t
*rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
v
,rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
v
,rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ь
$rnn/TensorArrayUnstack/strided_sliceStridedSlicernn/TensorArrayUnstack/Shape*rnn/TensorArrayUnstack/strided_slice/stack,rnn/TensorArrayUnstack/strided_slice/stack_1,rnn/TensorArrayUnstack/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
d
"rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
─
rnn/TensorArrayUnstack/rangeRange"rnn/TensorArrayUnstack/range/start$rnn/TensorArrayUnstack/strided_slice"rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:         *

Tidx0
ю
>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn/TensorArray_1rnn/TensorArrayUnstack/rangernn/transposernn/TensorArray_1:1*
T0* 
_class
loc:@rnn/transpose*
_output_shapes
: 
O
rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
[
rnn/MaximumMaximumrnn/Maximum/xrnn/strided_slice_1*
T0*
_output_shapes
: 
Y
rnn/MinimumMinimumrnn/strided_slice_1rnn/Maximum*
_output_shapes
: *
T0
]
rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
н
rnn/while/EnterEnterrnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *'

frame_namernn/while/while_context
Ь
rnn/while/Enter_1Enterrnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *'

frame_namernn/while/while_context
е
rnn/while/Enter_2Enterrnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *'

frame_namernn/while/while_context
└
rnn/while/Enter_3Enterrnn/GRUCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:         О*'

frame_namernn/while/while_context
n
rnn/while/MergeMergernn/while/Enterrnn/while/NextIteration*
T0*
N*
_output_shapes
: : 
t
rnn/while/Merge_1Mergernn/while/Enter_1rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
t
rnn/while/Merge_2Mergernn/while/Enter_2rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
Ж
rnn/while/Merge_3Mergernn/while/Enter_3rnn/while/NextIteration_3*
T0*
N**
_output_shapes
:         О: 
^
rnn/while/LessLessrnn/while/Mergernn/while/Less/Enter*
_output_shapes
: *
T0
к
rnn/while/Less/EnterEnterrnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *'

frame_namernn/while/while_context
d
rnn/while/Less_1Lessrnn/while/Merge_1rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
д
rnn/while/Less_1/EnterEnterrnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *'

frame_namernn/while/while_context
\
rnn/while/LogicalAnd
LogicalAndrnn/while/Lessrnn/while/Less_1*
_output_shapes
: 
L
rnn/while/LoopCondLoopCondrnn/while/LogicalAnd*
_output_shapes
: 
Ж
rnn/while/SwitchSwitchrnn/while/Mergernn/while/LoopCond*
_output_shapes
: : *
T0*"
_class
loc:@rnn/while/Merge
М
rnn/while/Switch_1Switchrnn/while/Merge_1rnn/while/LoopCond*
T0*$
_class
loc:@rnn/while/Merge_1*
_output_shapes
: : 
М
rnn/while/Switch_2Switchrnn/while/Merge_2rnn/while/LoopCond*
T0*$
_class
loc:@rnn/while/Merge_2*
_output_shapes
: : 
░
rnn/while/Switch_3Switchrnn/while/Merge_3rnn/while/LoopCond*
T0*$
_class
loc:@rnn/while/Merge_3*<
_output_shapes*
(:         О:         О
S
rnn/while/IdentityIdentityrnn/while/Switch:1*
T0*
_output_shapes
: 
W
rnn/while/Identity_1Identityrnn/while/Switch_1:1*
T0*
_output_shapes
: 
W
rnn/while/Identity_2Identityrnn/while/Switch_2:1*
T0*
_output_shapes
: 
i
rnn/while/Identity_3Identityrnn/while/Switch_3:1*
T0*(
_output_shapes
:         О
f
rnn/while/add/yConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Z
rnn/while/addAddrnn/while/Identityrnn/while/add/y*
_output_shapes
: *
T0
─
rnn/while/TensorArrayReadV3TensorArrayReadV3!rnn/while/TensorArrayReadV3/Enterrnn/while/Identity_1#rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:         
╣
!rnn/while/TensorArrayReadV3/EnterEnterrnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context
ф
#rnn/while/TensorArrayReadV3/Enter_1Enter>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *'

frame_namernn/while/while_context
╣
:rnn/gru_cell/gates/kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
valueB"    *
dtype0*
_output_shapes
:
л
8rnn/gru_cell/gates/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@rnn/gru_cell/gates/kernel*
valueB
 *nзо╜
л
8rnn/gru_cell/gates/kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
valueB
 *nзо=*
dtype0*
_output_shapes
: 
О
Brnn/gru_cell/gates/kernel/Initializer/random_uniform/RandomUniformRandomUniform:rnn/gru_cell/gates/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
ЭЬ*

seed *
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
seed2 
В
8rnn/gru_cell/gates/kernel/Initializer/random_uniform/subSub8rnn/gru_cell/gates/kernel/Initializer/random_uniform/max8rnn/gru_cell/gates/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
_output_shapes
: 
Ц
8rnn/gru_cell/gates/kernel/Initializer/random_uniform/mulMulBrnn/gru_cell/gates/kernel/Initializer/random_uniform/RandomUniform8rnn/gru_cell/gates/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel* 
_output_shapes
:
ЭЬ
И
4rnn/gru_cell/gates/kernel/Initializer/random_uniformAdd8rnn/gru_cell/gates/kernel/Initializer/random_uniform/mul8rnn/gru_cell/gates/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel* 
_output_shapes
:
ЭЬ
┐
rnn/gru_cell/gates/kernel
VariableV2*
shared_name *,
_class"
 loc:@rnn/gru_cell/gates/kernel*
	container *
shape:
ЭЬ*
dtype0* 
_output_shapes
:
ЭЬ
¤
 rnn/gru_cell/gates/kernel/AssignAssignrnn/gru_cell/gates/kernel4rnn/gru_cell/gates/kernel/Initializer/random_uniform*
use_locking(*
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
validate_shape(* 
_output_shapes
:
ЭЬ
p
rnn/gru_cell/gates/kernel/readIdentityrnn/gru_cell/gates/kernel*
T0* 
_output_shapes
:
ЭЬ
д
)rnn/gru_cell/gates/bias/Initializer/ConstConst**
_class 
loc:@rnn/gru_cell/gates/bias*
valueBЬ*  А?*
dtype0*
_output_shapes	
:Ь
▒
rnn/gru_cell/gates/bias
VariableV2**
_class 
loc:@rnn/gru_cell/gates/bias*
	container *
shape:Ь*
dtype0*
_output_shapes	
:Ь*
shared_name 
ч
rnn/gru_cell/gates/bias/AssignAssignrnn/gru_cell/gates/bias)rnn/gru_cell/gates/bias/Initializer/Const*
use_locking(*
T0**
_class 
loc:@rnn/gru_cell/gates/bias*
validate_shape(*
_output_shapes	
:Ь
g
rnn/gru_cell/gates/bias/readIdentityrnn/gru_cell/gates/bias*
_output_shapes	
:Ь*
T0
┴
>rnn/gru_cell/candidate/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
valueB"    
│
<rnn/gru_cell/candidate/kernel/Initializer/random_uniform/minConst*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
valueB
 *▀Ё╘╜*
dtype0*
_output_shapes
: 
│
<rnn/gru_cell/candidate/kernel/Initializer/random_uniform/maxConst*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
valueB
 *▀Ё╘=*
dtype0*
_output_shapes
: 
Ъ
Frnn/gru_cell/candidate/kernel/Initializer/random_uniform/RandomUniformRandomUniform>rnn/gru_cell/candidate/kernel/Initializer/random_uniform/shape*

seed *
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
seed2 *
dtype0* 
_output_shapes
:
ЭО
Т
<rnn/gru_cell/candidate/kernel/Initializer/random_uniform/subSub<rnn/gru_cell/candidate/kernel/Initializer/random_uniform/max<rnn/gru_cell/candidate/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
_output_shapes
: 
ж
<rnn/gru_cell/candidate/kernel/Initializer/random_uniform/mulMulFrnn/gru_cell/candidate/kernel/Initializer/random_uniform/RandomUniform<rnn/gru_cell/candidate/kernel/Initializer/random_uniform/sub*
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel* 
_output_shapes
:
ЭО
Ш
8rnn/gru_cell/candidate/kernel/Initializer/random_uniformAdd<rnn/gru_cell/candidate/kernel/Initializer/random_uniform/mul<rnn/gru_cell/candidate/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel* 
_output_shapes
:
ЭО
╟
rnn/gru_cell/candidate/kernel
VariableV2*
shared_name *0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
	container *
shape:
ЭО*
dtype0* 
_output_shapes
:
ЭО
Н
$rnn/gru_cell/candidate/kernel/AssignAssignrnn/gru_cell/candidate/kernel8rnn/gru_cell/candidate/kernel/Initializer/random_uniform*
use_locking(*
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
validate_shape(* 
_output_shapes
:
ЭО
x
"rnn/gru_cell/candidate/kernel/readIdentityrnn/gru_cell/candidate/kernel* 
_output_shapes
:
ЭО*
T0
м
-rnn/gru_cell/candidate/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:О*.
_class$
" loc:@rnn/gru_cell/candidate/bias*
valueBО*    
╣
rnn/gru_cell/candidate/bias
VariableV2*
	container *
shape:О*
dtype0*
_output_shapes	
:О*
shared_name *.
_class$
" loc:@rnn/gru_cell/candidate/bias
ў
"rnn/gru_cell/candidate/bias/AssignAssignrnn/gru_cell/candidate/bias-rnn/gru_cell/candidate/bias/Initializer/zeros*
T0*.
_class$
" loc:@rnn/gru_cell/candidate/bias*
validate_shape(*
_output_shapes	
:О*
use_locking(
o
 rnn/gru_cell/candidate/bias/readIdentityrnn/gru_cell/candidate/bias*
T0*
_output_shapes	
:О
u
rnn/while/gru_cell/concat/axisConst^rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
└
rnn/while/gru_cell/concatConcatV2rnn/while/TensorArrayReadV3rnn/while/Identity_3rnn/while/gru_cell/concat/axis*
N*(
_output_shapes
:         Э*

Tidx0*
T0
╕
rnn/while/gru_cell/MatMulMatMulrnn/while/gru_cell/concatrnn/while/gru_cell/MatMul/Enter*
T0*(
_output_shapes
:         Ь*
transpose_a( *
transpose_b( 
╩
rnn/while/gru_cell/MatMul/EnterEnterrnn/gru_cell/gates/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
ЭЬ*'

frame_namernn/while/while_context
м
rnn/while/gru_cell/BiasAddBiasAddrnn/while/gru_cell/MatMul rnn/while/gru_cell/BiasAdd/Enter*
T0*
data_formatNHWC*(
_output_shapes
:         Ь
─
 rnn/while/gru_cell/BiasAdd/EnterEnterrnn/gru_cell/gates/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:Ь*'

frame_namernn/while/while_context
t
rnn/while/gru_cell/SigmoidSigmoidrnn/while/gru_cell/BiasAdd*(
_output_shapes
:         Ь*
T0
o
rnn/while/gru_cell/ConstConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
y
"rnn/while/gru_cell/split/split_dimConst^rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
╣
rnn/while/gru_cell/splitSplit"rnn/while/gru_cell/split/split_dimrnn/while/gru_cell/Sigmoid*
T0*<
_output_shapes*
(:         О:         О*
	num_split
А
rnn/while/gru_cell/mulMulrnn/while/gru_cell/splitrnn/while/Identity_3*
T0*(
_output_shapes
:         О
w
 rnn/while/gru_cell/concat_1/axisConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╞
rnn/while/gru_cell/concat_1ConcatV2rnn/while/TensorArrayReadV3rnn/while/gru_cell/mul rnn/while/gru_cell/concat_1/axis*

Tidx0*
T0*
N*(
_output_shapes
:         Э
╛
rnn/while/gru_cell/MatMul_1MatMulrnn/while/gru_cell/concat_1!rnn/while/gru_cell/MatMul_1/Enter*
T0*(
_output_shapes
:         О*
transpose_a( *
transpose_b( 
╨
!rnn/while/gru_cell/MatMul_1/EnterEnter"rnn/gru_cell/candidate/kernel/read*
parallel_iterations * 
_output_shapes
:
ЭО*'

frame_namernn/while/while_context*
T0*
is_constant(
▓
rnn/while/gru_cell/BiasAdd_1BiasAddrnn/while/gru_cell/MatMul_1"rnn/while/gru_cell/BiasAdd_1/Enter*
data_formatNHWC*(
_output_shapes
:         О*
T0
╩
"rnn/while/gru_cell/BiasAdd_1/EnterEnter rnn/gru_cell/candidate/bias/read*
parallel_iterations *
_output_shapes	
:О*'

frame_namernn/while/while_context*
T0*
is_constant(
p
rnn/while/gru_cell/TanhTanhrnn/while/gru_cell/BiasAdd_1*
T0*(
_output_shapes
:         О
Д
rnn/while/gru_cell/mul_1Mulrnn/while/gru_cell/split:1rnn/while/Identity_3*(
_output_shapes
:         О*
T0
r
rnn/while/gru_cell/sub/xConst^rnn/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ж
rnn/while/gru_cell/subSubrnn/while/gru_cell/sub/xrnn/while/gru_cell/split:1*
T0*(
_output_shapes
:         О
Г
rnn/while/gru_cell/mul_2Mulrnn/while/gru_cell/subrnn/while/gru_cell/Tanh*
T0*(
_output_shapes
:         О
Д
rnn/while/gru_cell/addAddrnn/while/gru_cell/mul_1rnn/while/gru_cell/mul_2*(
_output_shapes
:         О*
T0
И
-rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn/while/Identity_1rnn/while/gru_cell/addrnn/while/Identity_2*
T0*)
_class
loc:@rnn/while/gru_cell/add*
_output_shapes
: 
Ї
3rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn/TensorArray*
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0*)
_class
loc:@rnn/while/gru_cell/add*
parallel_iterations 
h
rnn/while/add_1/yConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
rnn/while/add_1Addrnn/while/Identity_1rnn/while/add_1/y*
_output_shapes
: *
T0
X
rnn/while/NextIterationNextIterationrnn/while/add*
_output_shapes
: *
T0
\
rnn/while/NextIteration_1NextIterationrnn/while/add_1*
T0*
_output_shapes
: 
z
rnn/while/NextIteration_2NextIteration-rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
u
rnn/while/NextIteration_3NextIterationrnn/while/gru_cell/add*(
_output_shapes
:         О*
T0
I
rnn/while/ExitExitrnn/while/Switch*
T0*
_output_shapes
: 
M
rnn/while/Exit_1Exitrnn/while/Switch_1*
T0*
_output_shapes
: 
M
rnn/while/Exit_2Exitrnn/while/Switch_2*
_output_shapes
: *
T0
_
rnn/while/Exit_3Exitrnn/while/Switch_3*
T0*(
_output_shapes
:         О
Ъ
&rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn/TensorArrayrnn/while/Exit_2*"
_class
loc:@rnn/TensorArray*
_output_shapes
: 
Ж
 rnn/TensorArrayStack/range/startConst*"
_class
loc:@rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 
Ж
 rnn/TensorArrayStack/range/deltaConst*"
_class
loc:@rnn/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
ф
rnn/TensorArrayStack/rangeRange rnn/TensorArrayStack/range/start&rnn/TensorArrayStack/TensorArraySizeV3 rnn/TensorArrayStack/range/delta*#
_output_shapes
:         *

Tidx0*"
_class
loc:@rnn/TensorArray
Н
(rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn/TensorArrayrnn/TensorArrayStack/rangernn/while/Exit_2*%
element_shape:         О*"
_class
loc:@rnn/TensorArray*
dtype0*5
_output_shapes#
!:                  О
V
rnn/Const_1Const*
valueB:О*
dtype0*
_output_shapes
:
L

rnn/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
S
rnn/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
S
rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
n
rnn/range_1Rangernn/range_1/start
rnn/Rank_1rnn/range_1/delta*
_output_shapes
:*

Tidx0
f
rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
S
rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Й
rnn/concat_2ConcatV2rnn/concat_2/values_0rnn/range_1rnn/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
б
rnn/transpose_1	Transpose(rnn/TensorArrayStack/TensorArrayGatherV3rnn/concat_2*
T0*5
_output_shapes#
!:                  О*
Tperm0
R
GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
O
GatherV2/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Ь
GatherV2GatherV2rnn/transpose_1GatherV2/indicesGatherV2/axis*
Taxis0*
Tindices0*
Tparams0*(
_output_shapes
:         О
Я
-dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
_class
loc:@dense/kernel*
valueB"     
С
+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
valueB
 *0╛*
dtype0*
_output_shapes
: 
С
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *0>*
dtype0*
_output_shapes
: 
ц
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	О*

seed *
T0*
_class
loc:@dense/kernel*
seed2 
╬
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
с
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	О
╙
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	О
г
dense/kernel
VariableV2*
dtype0*
_output_shapes
:	О*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:	О
╚
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	О
v
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	О
И
dense/bias/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB*    *
dtype0*
_output_shapes
:
Х

dense/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@dense/bias
▓
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
k
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias*
_output_shapes
:
Л
dense/MatMulMatMulGatherV2dense/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
А
dense/BiasAddBiasAdddense/MatMuldense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
S
SoftmaxSoftmaxdense/BiasAdd*
T0*'
_output_shapes
:         
h
&softmax_cross_entropy_with_logits/RankConst*
dtype0*
_output_shapes
: *
value	B :
t
'softmax_cross_entropy_with_logits/ShapeShapedense/BiasAdd*
T0*
out_type0*
_output_shapes
:
j
(softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
v
)softmax_cross_entropy_with_logits/Shape_1Shapedense/BiasAdd*
T0*
out_type0*
_output_shapes
:
i
'softmax_cross_entropy_with_logits/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
а
%softmax_cross_entropy_with_logits/SubSub(softmax_cross_entropy_with_logits/Rank_1'softmax_cross_entropy_with_logits/Sub/y*
T0*
_output_shapes
: 
Ц
-softmax_cross_entropy_with_logits/Slice/beginPack%softmax_cross_entropy_with_logits/Sub*
T0*

axis *
N*
_output_shapes
:
v
,softmax_cross_entropy_with_logits/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ъ
'softmax_cross_entropy_with_logits/SliceSlice)softmax_cross_entropy_with_logits/Shape_1-softmax_cross_entropy_with_logits/Slice/begin,softmax_cross_entropy_with_logits/Slice/size*
Index0*
T0*
_output_shapes
:
Д
1softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
         *
dtype0*
_output_shapes
:
o
-softmax_cross_entropy_with_logits/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
∙
(softmax_cross_entropy_with_logits/concatConcatV21softmax_cross_entropy_with_logits/concat/values_0'softmax_cross_entropy_with_logits/Slice-softmax_cross_entropy_with_logits/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
╢
)softmax_cross_entropy_with_logits/ReshapeReshapedense/BiasAdd(softmax_cross_entropy_with_logits/concat*
T0*
Tshape0*0
_output_shapes
:                  
j
(softmax_cross_entropy_with_logits/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
v
)softmax_cross_entropy_with_logits/Shape_2ShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
k
)softmax_cross_entropy_with_logits/Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
д
'softmax_cross_entropy_with_logits/Sub_1Sub(softmax_cross_entropy_with_logits/Rank_2)softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
Ъ
/softmax_cross_entropy_with_logits/Slice_1/beginPack'softmax_cross_entropy_with_logits/Sub_1*
T0*

axis *
N*
_output_shapes
:
x
.softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Ё
)softmax_cross_entropy_with_logits/Slice_1Slice)softmax_cross_entropy_with_logits/Shape_2/softmax_cross_entropy_with_logits/Slice_1/begin.softmax_cross_entropy_with_logits/Slice_1/size*
Index0*
T0*
_output_shapes
:
Ж
3softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
         *
dtype0*
_output_shapes
:
q
/softmax_cross_entropy_with_logits/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Б
*softmax_cross_entropy_with_logits/concat_1ConcatV23softmax_cross_entropy_with_logits/concat_1/values_0)softmax_cross_entropy_with_logits/Slice_1/softmax_cross_entropy_with_logits/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
║
+softmax_cross_entropy_with_logits/Reshape_1ReshapePlaceholder_1*softmax_cross_entropy_with_logits/concat_1*
T0*
Tshape0*0
_output_shapes
:                  
ф
!softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits)softmax_cross_entropy_with_logits/Reshape+softmax_cross_entropy_with_logits/Reshape_1*
T0*?
_output_shapes-
+:         :                  
k
)softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
в
'softmax_cross_entropy_with_logits/Sub_2Sub&softmax_cross_entropy_with_logits/Rank)softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 
y
/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
Щ
.softmax_cross_entropy_with_logits/Slice_2/sizePack'softmax_cross_entropy_with_logits/Sub_2*
T0*

axis *
N*
_output_shapes
:
ў
)softmax_cross_entropy_with_logits/Slice_2Slice'softmax_cross_entropy_with_logits/Shape/softmax_cross_entropy_with_logits/Slice_2/begin.softmax_cross_entropy_with_logits/Slice_2/size*#
_output_shapes
:         *
Index0*
T0
└
+softmax_cross_entropy_with_logits/Reshape_2Reshape!softmax_cross_entropy_with_logits)softmax_cross_entropy_with_logits/Slice_2*#
_output_shapes
:         *
T0*
Tshape0
O
ConstConst*
dtype0*
_output_shapes
:*
valueB: 
~
MeanMean+softmax_cross_entropy_with_logits/Reshape_2Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
~
ArgMaxArgMaxdense/BiasAddArgMax/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:         
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
В
ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:         *

Tidx0
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:         
P
CastCastEqual*

SrcT0
*#
_output_shapes
:         *

DstT0
Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
[
Mean_1MeanCastConst_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
S
gradients/f_countConst*
dtype0*
_output_shapes
: *
value	B : 
з
gradients/f_count_1Entergradients/f_count*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *'

frame_namernn/while/while_context
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
T0*
N*
_output_shapes
: : 
b
gradients/SwitchSwitchgradients/Mergernn/while/LoopCond*
T0*
_output_shapes
: : 
f
gradients/Add/yConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
T0*
_output_shapes
: 
═
gradients/NextIterationNextIterationgradients/Add[^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2@^gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/StackPushV2>^gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/StackPushV2B^gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/StackPushV2H^gradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPushV2J^gradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPushV2_1>^gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPushV2@^gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPushV2_1J^gradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_18^gradients/rnn/while/gru_cell/mul_1_grad/Mul/StackPushV2:^gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/StackPushV2J^gradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_18^gradients/rnn/while/gru_cell/mul_2_grad/Mul/StackPushV2:^gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/StackPushV2H^gradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPushV2J^gradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPushV2_18^gradients/rnn/while/gru_cell/mul_grad/Mul_1/StackPushV2H^gradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/StackPushV2*
_output_shapes
: *
T0
N
gradients/f_count_2Exitgradients/Switch*
T0*
_output_shapes
: 
S
gradients/b_countConst*
value	B :*
dtype0*
_output_shapes
: 
│
gradients/b_count_1Entergradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *1

frame_name#!gradients/rnn/while/while_context
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 
x
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
║
gradients/GreaterEqual/EnterEntergradients/b_count*
parallel_iterations *
_output_shapes
: *1

frame_name#!gradients/rnn/while/while_context*
T0*
is_constant(
O
gradients/b_count_2LoopCondgradients/GreaterEqual*
_output_shapes
: 
g
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*
_output_shapes
: : *
T0
i
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
▓
gradients/NextIteration_1NextIterationgradients/SubV^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
_output_shapes
: *
T0
P
gradients/b_count_3Exitgradients/Switch_1*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
М
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
Д
gradients/Mean_grad/ShapeShape+softmax_cross_entropy_with_logits/Reshape_2*
T0*
out_type0*
_output_shapes
:
Ш
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:         *

Tmultiples0*
T0
Ж
gradients/Mean_grad/Shape_1Shape+softmax_cross_entropy_with_logits/Reshape_2*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ц
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
В
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
А
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
И
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:         
б
@gradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShape!softmax_cross_entropy_with_logits*
T0*
out_type0*
_output_shapes
:
ш
Bgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truediv@gradients/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
Б
gradients/zeros_like	ZerosLike#softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:                  
К
?gradients/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
         
М
;gradients/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDimsBgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshape?gradients/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
╪
4gradients/softmax_cross_entropy_with_logits_grad/mulMul;gradients/softmax_cross_entropy_with_logits_grad/ExpandDims#softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:                  
п
;gradients/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmax)softmax_cross_entropy_with_logits/Reshape*0
_output_shapes
:                  *
T0
│
4gradients/softmax_cross_entropy_with_logits_grad/NegNeg;gradients/softmax_cross_entropy_with_logits_grad/LogSoftmax*
T0*0
_output_shapes
:                  
М
Agradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
Р
=gradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDimsBgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*
T0*'
_output_shapes
:         *

Tdim0
э
6gradients/softmax_cross_entropy_with_logits_grad/mul_1Mul=gradients/softmax_cross_entropy_with_logits_grad/ExpandDims_14gradients/softmax_cross_entropy_with_logits_grad/Neg*
T0*0
_output_shapes
:                  
╣
Agradients/softmax_cross_entropy_with_logits_grad/tuple/group_depsNoOp5^gradients/softmax_cross_entropy_with_logits_grad/mul7^gradients/softmax_cross_entropy_with_logits_grad/mul_1
╙
Igradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependencyIdentity4gradients/softmax_cross_entropy_with_logits_grad/mulB^gradients/softmax_cross_entropy_with_logits_grad/tuple/group_deps*0
_output_shapes
:                  *
T0*G
_class=
;9loc:@gradients/softmax_cross_entropy_with_logits_grad/mul
┘
Kgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency_1Identity6gradients/softmax_cross_entropy_with_logits_grad/mul_1B^gradients/softmax_cross_entropy_with_logits_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/softmax_cross_entropy_with_logits_grad/mul_1*0
_output_shapes
:                  
Л
>gradients/softmax_cross_entropy_with_logits/Reshape_grad/ShapeShapedense/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ц
@gradients/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshapeIgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency>gradients/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
╡
(gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad@gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
г
-gradients/dense/BiasAdd_grad/tuple/group_depsNoOp)^gradients/dense/BiasAdd_grad/BiasAddGradA^gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshape
║
5gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentity@gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshape.^gradients/dense/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*'
_output_shapes
:         
 
7gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/dense/BiasAdd_grad/BiasAddGrad.^gradients/dense/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
╧
"gradients/dense/MatMul_grad/MatMulMatMul5gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
T0*(
_output_shapes
:         О*
transpose_a( *
transpose_b(
┐
$gradients/dense/MatMul_grad/MatMul_1MatMulGatherV25gradients/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	О*
transpose_a(*
transpose_b( 
А
,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
¤
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul*(
_output_shapes
:         О
·
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1*
_output_shapes
:	О
Р
gradients/GatherV2_grad/ShapeShapernn/transpose_1*
_output_shapes
:*
T0*"
_class
loc:@rnn/transpose_1*
out_type0	
Ю
gradients/GatherV2_grad/ToInt32Castgradients/GatherV2_grad/Shape*

SrcT0	*"
_class
loc:@rnn/transpose_1*
_output_shapes
:*

DstT0
^
gradients/GatherV2_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
h
&gradients/GatherV2_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
з
"gradients/GatherV2_grad/ExpandDims
ExpandDimsgradients/GatherV2_grad/Size&gradients/GatherV2_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
u
+gradients/GatherV2_grad/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
~
-gradients/GatherV2_grad/strided_slice/stack_1PackGatherV2/axis*
T0*

axis *
N*
_output_shapes
:
w
-gradients/GatherV2_grad/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ў
%gradients/GatherV2_grad/strided_sliceStridedSlicegradients/GatherV2_grad/ToInt32+gradients/GatherV2_grad/strided_slice/stack-gradients/GatherV2_grad/strided_slice/stack_1-gradients/GatherV2_grad/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0*
shrink_axis_mask 
`
gradients/GatherV2_grad/Size_1Const*
value	B :*
dtype0*
_output_shapes
: 
~
-gradients/GatherV2_grad/strided_slice_1/stackPackGatherV2/axis*
T0*

axis *
N*
_output_shapes
:
y
/gradients/GatherV2_grad/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
y
/gradients/GatherV2_grad/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
 
'gradients/GatherV2_grad/strided_slice_1StridedSlicegradients/GatherV2_grad/ToInt32-gradients/GatherV2_grad/strided_slice_1/stack/gradients/GatherV2_grad/strided_slice_1/stack_1/gradients/GatherV2_grad/strided_slice_1/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0
w
-gradients/GatherV2_grad/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
y
/gradients/GatherV2_grad/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
y
/gradients/GatherV2_grad/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
З
'gradients/GatherV2_grad/strided_slice_2StridedSlice'gradients/GatherV2_grad/strided_slice_1-gradients/GatherV2_grad/strided_slice_2/stack/gradients/GatherV2_grad/strided_slice_2/stack_1/gradients/GatherV2_grad/strided_slice_2/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0
`
gradients/GatherV2_grad/Size_2Const*
value	B :*
dtype0*
_output_shapes
: 
e
#gradients/GatherV2_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
e
#gradients/GatherV2_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
╕
gradients/GatherV2_grad/rangeRange#gradients/GatherV2_grad/range/startgradients/GatherV2_grad/Size_1#gradients/GatherV2_grad/range/delta*
_output_shapes
:*

Tidx0
_
gradients/GatherV2_grad/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
В
gradients/GatherV2_grad/addAddgradients/GatherV2_grad/Size_1gradients/GatherV2_grad/add/y*
T0*
_output_shapes
: 
a
gradients/GatherV2_grad/add_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ж
gradients/GatherV2_grad/add_1Addgradients/GatherV2_grad/Size_1gradients/GatherV2_grad/add_1/y*
T0*
_output_shapes
: 
Д
gradients/GatherV2_grad/add_2Addgradients/GatherV2_grad/add_1gradients/GatherV2_grad/Size_2*
T0*
_output_shapes
: 
g
%gradients/GatherV2_grad/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
╝
gradients/GatherV2_grad/range_1Rangegradients/GatherV2_grad/addgradients/GatherV2_grad/add_2%gradients/GatherV2_grad/range_1/delta*#
_output_shapes
:         *

Tidx0
e
#gradients/GatherV2_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
¤
gradients/GatherV2_grad/concatConcatV2%gradients/GatherV2_grad/strided_slice"gradients/GatherV2_grad/ExpandDims'gradients/GatherV2_grad/strided_slice_2#gradients/GatherV2_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
╓
gradients/GatherV2_grad/ReshapeReshape4gradients/dense/MatMul_grad/tuple/control_dependencygradients/GatherV2_grad/concat*=
_output_shapes+
):'                           *
T0*
Tshape0
Х
!gradients/GatherV2_grad/Reshape_1ReshapeGatherV2/indices"gradients/GatherV2_grad/ExpandDims*
T0*
Tshape0*
_output_shapes
:
Л
)gradients/GatherV2_grad/concat_1/values_0Packgradients/GatherV2_grad/Size_1*
T0*

axis *
N*
_output_shapes
:
g
%gradients/GatherV2_grad/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Б
 gradients/GatherV2_grad/concat_1ConcatV2)gradients/GatherV2_grad/concat_1/values_0gradients/GatherV2_grad/rangegradients/GatherV2_grad/range_1%gradients/GatherV2_grad/concat_1/axis*
T0*
N*#
_output_shapes
:         *

Tidx0
╞
!gradients/GatherV2_grad/transpose	Transposegradients/GatherV2_grad/Reshape gradients/GatherV2_grad/concat_1*
T0*=
_output_shapes+
):'                           *
Tperm0
a
gradients/GatherV2_grad/add_3/yConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/GatherV2_grad/add_3AddGatherV2/axisgradients/GatherV2_grad/add_3/y*
T0*
_output_shapes
: 
~
-gradients/GatherV2_grad/strided_slice_3/stackPackGatherV2/axis*
T0*

axis *
N*
_output_shapes
:
Р
/gradients/GatherV2_grad/strided_slice_3/stack_1Packgradients/GatherV2_grad/add_3*
T0*

axis *
N*
_output_shapes
:
y
/gradients/GatherV2_grad/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
√
'gradients/GatherV2_grad/strided_slice_3StridedSlicegradients/GatherV2_grad/ToInt32-gradients/GatherV2_grad/strided_slice_3/stack/gradients/GatherV2_grad/strided_slice_3/stack_1/gradients/GatherV2_grad/strided_slice_3/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Ы
*gradients/GatherV2_grad/UnsortedSegmentSumUnsortedSegmentSum!gradients/GatherV2_grad/transpose!gradients/GatherV2_grad/Reshape_1'gradients/GatherV2_grad/strided_slice_3*
Tnumsegments0*
Tindices0*
T0*=
_output_shapes+
):'                           
a
gradients/GatherV2_grad/add_4/yConst*
value	B :*
dtype0*
_output_shapes
: 
Й
gradients/GatherV2_grad/add_4Addgradients/GatherV2_grad/rangegradients/GatherV2_grad/add_4/y*
T0*
_output_shapes
:
s
)gradients/GatherV2_grad/concat_2/values_1Const*
dtype0*
_output_shapes
:*
valueB: 
g
%gradients/GatherV2_grad/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Б
 gradients/GatherV2_grad/concat_2ConcatV2gradients/GatherV2_grad/add_4)gradients/GatherV2_grad/concat_2/values_1gradients/GatherV2_grad/range_1%gradients/GatherV2_grad/concat_2/axis*
T0*
N*#
_output_shapes
:         *

Tidx0
╦
#gradients/GatherV2_grad/transpose_1	Transpose*gradients/GatherV2_grad/UnsortedSegmentSum gradients/GatherV2_grad/concat_2*
T0*5
_output_shapes#
!:                  О*
Tperm0
x
0gradients/rnn/transpose_1_grad/InvertPermutationInvertPermutationrnn/concat_2*
T0*
_output_shapes
:
┘
(gradients/rnn/transpose_1_grad/transpose	Transpose#gradients/GatherV2_grad/transpose_10gradients/rnn/transpose_1_grad/InvertPermutation*
T0*5
_output_shapes#
!:                  О*
Tperm0
ъ
Ygradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn/TensorArrayrnn/while/Exit_2*"
_class
loc:@rnn/TensorArray*
source	gradients*
_output_shapes

:: 
Ф
Ugradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityrnn/while/Exit_2Z^gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*"
_class
loc:@rnn/TensorArray
Р
_gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3Ygradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3rnn/TensorArrayStack/range(gradients/rnn/transpose_1_grad/transposeUgradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
h
gradients/zeros_like_1	ZerosLikernn/while/Exit_3*
T0*(
_output_shapes
:         О
Т
&gradients/rnn/while/Exit_2_grad/b_exitEnter_gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *1

frame_name#!gradients/rnn/while/while_context
█
&gradients/rnn/while/Exit_3_grad/b_exitEntergradients/zeros_like_1*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:         О*1

frame_name#!gradients/rnn/while/while_context
║
*gradients/rnn/while/Switch_2_grad/b_switchMerge&gradients/rnn/while/Exit_2_grad/b_exit1gradients/rnn/while/Switch_2_grad_1/NextIteration*
T0*
N*
_output_shapes
: : 
╠
*gradients/rnn/while/Switch_3_grad/b_switchMerge&gradients/rnn/while/Exit_3_grad/b_exit1gradients/rnn/while/Switch_3_grad_1/NextIteration*
N**
_output_shapes
:         О: *
T0
╘
'gradients/rnn/while/Merge_2_grad/SwitchSwitch*gradients/rnn/while/Switch_2_grad/b_switchgradients/b_count_2*
_output_shapes
: : *
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch
c
1gradients/rnn/while/Merge_2_grad/tuple/group_depsNoOp(^gradients/rnn/while/Merge_2_grad/Switch
В
9gradients/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity'gradients/rnn/while/Merge_2_grad/Switch2^gradients/rnn/while/Merge_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
Ж
;gradients/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity)gradients/rnn/while/Merge_2_grad/Switch:12^gradients/rnn/while/Merge_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
°
'gradients/rnn/while/Merge_3_grad/SwitchSwitch*gradients/rnn/while/Switch_3_grad/b_switchgradients/b_count_2*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch*<
_output_shapes*
(:         О:         О
c
1gradients/rnn/while/Merge_3_grad/tuple/group_depsNoOp(^gradients/rnn/while/Merge_3_grad/Switch
Ф
9gradients/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity'gradients/rnn/while/Merge_3_grad/Switch2^gradients/rnn/while/Merge_3_grad/tuple/group_deps*(
_output_shapes
:         О*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch
Ш
;gradients/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity)gradients/rnn/while/Merge_3_grad/Switch:12^gradients/rnn/while/Merge_3_grad/tuple/group_deps*(
_output_shapes
:         О*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch
Й
%gradients/rnn/while/Enter_2_grad/ExitExit9gradients/rnn/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 
Ы
%gradients/rnn/while/Enter_3_grad/ExitExit9gradients/rnn/while/Merge_3_grad/tuple/control_dependency*
T0*(
_output_shapes
:         О
Ў
^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3dgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter;gradients/rnn/while/Merge_2_grad/tuple/control_dependency_1*)
_class
loc:@rnn/while/gru_cell/add*
source	gradients*
_output_shapes

:: 
п
dgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn/TensorArray*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0*)
_class
loc:@rnn/while/gru_cell/add*
parallel_iterations *
is_constant(
╨
Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity;gradients/rnn/while/Merge_2_grad/tuple/control_dependency_1_^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*)
_class
loc:@rnn/while/gru_cell/add*
_output_shapes
: 
▒
Ngradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3Ygradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*(
_output_shapes
:         О
╚
Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*'
_class
loc:@rnn/while/Identity_1*
valueB :
         *
dtype0*
_output_shapes
: 
Э
Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*
	elem_type0*'
_class
loc:@rnn/while/Identity_1*

stack_name *
_output_shapes
:
п
Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterTgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context
Щ
Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn/while/Identity_1^gradients/Add*
_output_shapes
: *
swap_memory( *
T0
Б
Ygradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2_gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub*
_output_shapes
: *
	elem_type0
─
_gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterTgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0*
is_constant(
╚
Ugradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerZ^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2?^gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/StackPopV2=^gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/StackPopV2A^gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/StackPopV2G^gradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPopV2I^gradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPopV2_1=^gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPopV2?^gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPopV2_1I^gradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_17^gradients/rnn/while/gru_cell/mul_1_grad/Mul/StackPopV29^gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/StackPopV2I^gradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_17^gradients/rnn/while/gru_cell/mul_2_grad/Mul/StackPopV29^gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/StackPopV2G^gradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPopV2I^gradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPopV2_17^gradients/rnn/while/gru_cell/mul_grad/Mul_1/StackPopV2G^gradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/StackPopV2
ф
Mgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOp<^gradients/rnn/while/Merge_2_grad/tuple/control_dependency_1O^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
Ч
Ugradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityNgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3N^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*(
_output_shapes
:         О
╨
Wgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1Identity;gradients/rnn/while/Merge_2_grad/tuple/control_dependency_1N^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
е
gradients/AddNAddN;gradients/rnn/while/Merge_3_grad/tuple/control_dependency_1Ugradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch*
N*(
_output_shapes
:         О
Г
+gradients/rnn/while/gru_cell/add_grad/ShapeShapernn/while/gru_cell/mul_1*
T0*
out_type0*
_output_shapes
:
Е
-gradients/rnn/while/gru_cell/add_grad/Shape_1Shapernn/while/gru_cell/mul_2*
T0*
out_type0*
_output_shapes
:
г
;gradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPopV2Hgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
╠
Agradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/ConstConst*>
_class4
20loc:@gradients/rnn/while/gru_cell/add_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
О
Agradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/f_accStackV2Agradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*>
_class4
20loc:@gradients/rnn/while/gru_cell/add_grad/Shape
Й
Agradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/EnterEnterAgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context*
T0*
is_constant(
О
Ggradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/Enter+gradients/rnn/while/gru_cell/add_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
▀
Fgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Lgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
Ю
Lgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterAgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context
╨
Cgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/Const_1Const*@
_class6
42loc:@gradients/rnn/while/gru_cell/add_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
Ф
Cgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/f_acc_1StackV2Cgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/Const_1*

stack_name *
_output_shapes
:*
	elem_type0*@
_class6
42loc:@gradients/rnn/while/gru_cell/add_grad/Shape_1
Н
Cgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/Enter_1EnterCgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context
Ф
Igradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Cgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/Enter_1-gradients/rnn/while/gru_cell/add_grad/Shape_1^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
у
Hgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Ngradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
в
Ngradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterCgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context
╜
)gradients/rnn/while/gru_cell/add_grad/SumSumgradients/AddN;gradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ь
-gradients/rnn/while/gru_cell/add_grad/ReshapeReshape)gradients/rnn/while/gru_cell/add_grad/SumFgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPopV2*(
_output_shapes
:         О*
T0*
Tshape0
┴
+gradients/rnn/while/gru_cell/add_grad/Sum_1Sumgradients/AddN=gradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Є
/gradients/rnn/while/gru_cell/add_grad/Reshape_1Reshape+gradients/rnn/while/gru_cell/add_grad/Sum_1Hgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         О
а
6gradients/rnn/while/gru_cell/add_grad/tuple/group_depsNoOp.^gradients/rnn/while/gru_cell/add_grad/Reshape0^gradients/rnn/while/gru_cell/add_grad/Reshape_1
з
>gradients/rnn/while/gru_cell/add_grad/tuple/control_dependencyIdentity-gradients/rnn/while/gru_cell/add_grad/Reshape7^gradients/rnn/while/gru_cell/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn/while/gru_cell/add_grad/Reshape*(
_output_shapes
:         О
н
@gradients/rnn/while/gru_cell/add_grad/tuple/control_dependency_1Identity/gradients/rnn/while/gru_cell/add_grad/Reshape_17^gradients/rnn/while/gru_cell/add_grad/tuple/group_deps*(
_output_shapes
:         О*
T0*B
_class8
64loc:@gradients/rnn/while/gru_cell/add_grad/Reshape_1
З
-gradients/rnn/while/gru_cell/mul_1_grad/ShapeShapernn/while/gru_cell/split:1*
_output_shapes
:*
T0*
out_type0
Г
/gradients/rnn/while/gru_cell/mul_1_grad/Shape_1Shapernn/while/Identity_3*
T0*
out_type0*
_output_shapes
:
й
=gradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
╨
Cgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*@
_class6
42loc:@gradients/rnn/while/gru_cell/mul_1_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
Ф
Cgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/Const*@
_class6
42loc:@gradients/rnn/while/gru_cell/mul_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
Н
Cgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context
Ф
Igradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/Enter-gradients/rnn/while/gru_cell/mul_1_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
у
Hgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
в
Ngradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context
╘
Egradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*B
_class8
64loc:@gradients/rnn/while/gru_cell/mul_1_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
Ъ
Egradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/Const_1*

stack_name *
_output_shapes
:*
	elem_type0*B
_class8
64loc:@gradients/rnn/while/gru_cell/mul_1_grad/Shape_1
С
Egradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context
Ъ
Kgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/Enter_1/gradients/rnn/while/gru_cell/mul_1_grad/Shape_1^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
ч
Jgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
ж
Pgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context
▌
+gradients/rnn/while/gru_cell/mul_1_grad/MulMul>gradients/rnn/while/gru_cell/add_grad/tuple/control_dependency6gradients/rnn/while/gru_cell/mul_1_grad/Mul/StackPopV2*(
_output_shapes
:         О*
T0
е
1gradients/rnn/while/gru_cell/mul_1_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *'
_class
loc:@rnn/while/Identity_3*
valueB :
         
╫
1gradients/rnn/while/gru_cell/mul_1_grad/Mul/f_accStackV21gradients/rnn/while/gru_cell/mul_1_grad/Mul/Const*

stack_name *
_output_shapes
:*
	elem_type0*'
_class
loc:@rnn/while/Identity_3
щ
1gradients/rnn/while/gru_cell/mul_1_grad/Mul/EnterEnter1gradients/rnn/while/gru_cell/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context
х
7gradients/rnn/while/gru_cell/mul_1_grad/Mul/StackPushV2StackPushV21gradients/rnn/while/gru_cell/mul_1_grad/Mul/Enterrnn/while/Identity_3^gradients/Add*
T0*(
_output_shapes
:         О*
swap_memory( 
═
6gradients/rnn/while/gru_cell/mul_1_grad/Mul/StackPopV2
StackPopV2<gradients/rnn/while/gru_cell/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub*(
_output_shapes
:         О*
	elem_type0
■
<gradients/rnn/while/gru_cell/mul_1_grad/Mul/StackPopV2/EnterEnter1gradients/rnn/while/gru_cell/mul_1_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0*
is_constant(
▐
+gradients/rnn/while/gru_cell/mul_1_grad/SumSum+gradients/rnn/while/gru_cell/mul_1_grad/Mul=gradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Є
/gradients/rnn/while/gru_cell/mul_1_grad/ReshapeReshape+gradients/rnn/while/gru_cell/mul_1_grad/SumHgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         О
с
-gradients/rnn/while/gru_cell/mul_1_grad/Mul_1Mul8gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/StackPopV2>gradients/rnn/while/gru_cell/add_grad/tuple/control_dependency*
T0*(
_output_shapes
:         О
л
3gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/ConstConst*+
_class!
loc:@rnn/while/gru_cell/split*
valueB :
         *
dtype0*
_output_shapes
: 
▀
3gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/f_accStackV23gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/Const*+
_class!
loc:@rnn/while/gru_cell/split*

stack_name *
_output_shapes
:*
	elem_type0
э
3gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/EnterEnter3gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context*
T0*
is_constant(
я
9gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/StackPushV2StackPushV23gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/Enterrnn/while/gru_cell/split:1^gradients/Add*
T0*(
_output_shapes
:         О*
swap_memory( 
╤
8gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2>gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*(
_output_shapes
:         О
В
>gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnter3gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context
ф
-gradients/rnn/while/gru_cell/mul_1_grad/Sum_1Sum-gradients/rnn/while/gru_cell/mul_1_grad/Mul_1?gradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
°
1gradients/rnn/while/gru_cell/mul_1_grad/Reshape_1Reshape-gradients/rnn/while/gru_cell/mul_1_grad/Sum_1Jgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         О
ж
8gradients/rnn/while/gru_cell/mul_1_grad/tuple/group_depsNoOp0^gradients/rnn/while/gru_cell/mul_1_grad/Reshape2^gradients/rnn/while/gru_cell/mul_1_grad/Reshape_1
п
@gradients/rnn/while/gru_cell/mul_1_grad/tuple/control_dependencyIdentity/gradients/rnn/while/gru_cell/mul_1_grad/Reshape9^gradients/rnn/while/gru_cell/mul_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn/while/gru_cell/mul_1_grad/Reshape*(
_output_shapes
:         О
╡
Bgradients/rnn/while/gru_cell/mul_1_grad/tuple/control_dependency_1Identity1gradients/rnn/while/gru_cell/mul_1_grad/Reshape_19^gradients/rnn/while/gru_cell/mul_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/while/gru_cell/mul_1_grad/Reshape_1*(
_output_shapes
:         О
Г
-gradients/rnn/while/gru_cell/mul_2_grad/ShapeShapernn/while/gru_cell/sub*
T0*
out_type0*
_output_shapes
:
Ж
/gradients/rnn/while/gru_cell/mul_2_grad/Shape_1Shapernn/while/gru_cell/Tanh*
T0*
out_type0*
_output_shapes
:
й
=gradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
╨
Cgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*@
_class6
42loc:@gradients/rnn/while/gru_cell/mul_2_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
Ф
Cgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/Const*@
_class6
42loc:@gradients/rnn/while/gru_cell/mul_2_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
Н
Cgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context
Ф
Igradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/Enter-gradients/rnn/while/gru_cell/mul_2_grad/Shape^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
у
Hgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
в
Ngradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context
╘
Egradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *B
_class8
64loc:@gradients/rnn/while/gru_cell/mul_2_grad/Shape_1*
valueB :
         
Ъ
Egradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/Const_1*B
_class8
64loc:@gradients/rnn/while/gru_cell/mul_2_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
С
Egradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context*
T0*
is_constant(
Ъ
Kgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/Enter_1/gradients/rnn/while/gru_cell/mul_2_grad/Shape_1^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
ч
Jgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
ж
Pgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0*
is_constant(
▀
+gradients/rnn/while/gru_cell/mul_2_grad/MulMul@gradients/rnn/while/gru_cell/add_grad/tuple/control_dependency_16gradients/rnn/while/gru_cell/mul_2_grad/Mul/StackPopV2*
T0*(
_output_shapes
:         О
и
1gradients/rnn/while/gru_cell/mul_2_grad/Mul/ConstConst*
dtype0*
_output_shapes
: **
_class 
loc:@rnn/while/gru_cell/Tanh*
valueB :
         
┌
1gradients/rnn/while/gru_cell/mul_2_grad/Mul/f_accStackV21gradients/rnn/while/gru_cell/mul_2_grad/Mul/Const**
_class 
loc:@rnn/while/gru_cell/Tanh*

stack_name *
_output_shapes
:*
	elem_type0
щ
1gradients/rnn/while/gru_cell/mul_2_grad/Mul/EnterEnter1gradients/rnn/while/gru_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context
ш
7gradients/rnn/while/gru_cell/mul_2_grad/Mul/StackPushV2StackPushV21gradients/rnn/while/gru_cell/mul_2_grad/Mul/Enterrnn/while/gru_cell/Tanh^gradients/Add*
T0*(
_output_shapes
:         О*
swap_memory( 
═
6gradients/rnn/while/gru_cell/mul_2_grad/Mul/StackPopV2
StackPopV2<gradients/rnn/while/gru_cell/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub*(
_output_shapes
:         О*
	elem_type0
■
<gradients/rnn/while/gru_cell/mul_2_grad/Mul/StackPopV2/EnterEnter1gradients/rnn/while/gru_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context
▐
+gradients/rnn/while/gru_cell/mul_2_grad/SumSum+gradients/rnn/while/gru_cell/mul_2_grad/Mul=gradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Є
/gradients/rnn/while/gru_cell/mul_2_grad/ReshapeReshape+gradients/rnn/while/gru_cell/mul_2_grad/SumHgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*(
_output_shapes
:         О*
T0*
Tshape0
у
-gradients/rnn/while/gru_cell/mul_2_grad/Mul_1Mul8gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/StackPopV2@gradients/rnn/while/gru_cell/add_grad/tuple/control_dependency_1*(
_output_shapes
:         О*
T0
й
3gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/ConstConst*)
_class
loc:@rnn/while/gru_cell/sub*
valueB :
         *
dtype0*
_output_shapes
: 
▌
3gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/f_accStackV23gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/Const*
	elem_type0*)
_class
loc:@rnn/while/gru_cell/sub*

stack_name *
_output_shapes
:
э
3gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/EnterEnter3gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context
ы
9gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/StackPushV2StackPushV23gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/Enterrnn/while/gru_cell/sub^gradients/Add*(
_output_shapes
:         О*
swap_memory( *
T0
╤
8gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2>gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub*(
_output_shapes
:         О*
	elem_type0
В
>gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnter3gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context
ф
-gradients/rnn/while/gru_cell/mul_2_grad/Sum_1Sum-gradients/rnn/while/gru_cell/mul_2_grad/Mul_1?gradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
°
1gradients/rnn/while/gru_cell/mul_2_grad/Reshape_1Reshape-gradients/rnn/while/gru_cell/mul_2_grad/Sum_1Jgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*(
_output_shapes
:         О*
T0*
Tshape0
ж
8gradients/rnn/while/gru_cell/mul_2_grad/tuple/group_depsNoOp0^gradients/rnn/while/gru_cell/mul_2_grad/Reshape2^gradients/rnn/while/gru_cell/mul_2_grad/Reshape_1
п
@gradients/rnn/while/gru_cell/mul_2_grad/tuple/control_dependencyIdentity/gradients/rnn/while/gru_cell/mul_2_grad/Reshape9^gradients/rnn/while/gru_cell/mul_2_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn/while/gru_cell/mul_2_grad/Reshape*(
_output_shapes
:         О
╡
Bgradients/rnn/while/gru_cell/mul_2_grad/tuple/control_dependency_1Identity1gradients/rnn/while/gru_cell/mul_2_grad/Reshape_19^gradients/rnn/while/gru_cell/mul_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/while/gru_cell/mul_2_grad/Reshape_1*(
_output_shapes
:         О
╝
1gradients/rnn/while/Switch_2_grad_1/NextIterationNextIterationWgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
~
+gradients/rnn/while/gru_cell/sub_grad/ShapeConst^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
З
-gradients/rnn/while/gru_cell/sub_grad/Shape_1Shapernn/while/gru_cell/split:1*
T0*
out_type0*
_output_shapes
:
Ж
;gradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients/rnn/while/gru_cell/sub_grad/ShapeFgradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/StackPopV2*
T0*2
_output_shapes 
:         :         
╬
Agradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/ConstConst*@
_class6
42loc:@gradients/rnn/while/gru_cell/sub_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
Р
Agradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/f_accStackV2Agradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/Const*@
_class6
42loc:@gradients/rnn/while/gru_cell/sub_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
Й
Agradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/EnterEnterAgradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context
Р
Ggradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/Enter-gradients/rnn/while/gru_cell/sub_grad/Shape_1^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
▀
Fgradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Lgradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
Ю
Lgradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/StackPopV2/EnterEnterAgradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context
я
)gradients/rnn/while/gru_cell/sub_grad/SumSum@gradients/rnn/while/gru_cell/mul_2_grad/tuple/control_dependency;gradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┐
-gradients/rnn/while/gru_cell/sub_grad/ReshapeReshape)gradients/rnn/while/gru_cell/sub_grad/Sum+gradients/rnn/while/gru_cell/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
є
+gradients/rnn/while/gru_cell/sub_grad/Sum_1Sum@gradients/rnn/while/gru_cell/mul_2_grad/tuple/control_dependency=gradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
А
)gradients/rnn/while/gru_cell/sub_grad/NegNeg+gradients/rnn/while/gru_cell/sub_grad/Sum_1*
T0*
_output_shapes
:
ю
/gradients/rnn/while/gru_cell/sub_grad/Reshape_1Reshape)gradients/rnn/while/gru_cell/sub_grad/NegFgradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         О
а
6gradients/rnn/while/gru_cell/sub_grad/tuple/group_depsNoOp.^gradients/rnn/while/gru_cell/sub_grad/Reshape0^gradients/rnn/while/gru_cell/sub_grad/Reshape_1
Х
>gradients/rnn/while/gru_cell/sub_grad/tuple/control_dependencyIdentity-gradients/rnn/while/gru_cell/sub_grad/Reshape7^gradients/rnn/while/gru_cell/sub_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn/while/gru_cell/sub_grad/Reshape*
_output_shapes
: 
н
@gradients/rnn/while/gru_cell/sub_grad/tuple/control_dependency_1Identity/gradients/rnn/while/gru_cell/sub_grad/Reshape_17^gradients/rnn/while/gru_cell/sub_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn/while/gru_cell/sub_grad/Reshape_1*(
_output_shapes
:         О
ъ
/gradients/rnn/while/gru_cell/Tanh_grad/TanhGradTanhGrad6gradients/rnn/while/gru_cell/mul_2_grad/Mul/StackPopV2Bgradients/rnn/while/gru_cell/mul_2_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         О
┤
7gradients/rnn/while/gru_cell/BiasAdd_1_grad/BiasAddGradBiasAddGrad/gradients/rnn/while/gru_cell/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:О*
T0
░
<gradients/rnn/while/gru_cell/BiasAdd_1_grad/tuple/group_depsNoOp8^gradients/rnn/while/gru_cell/BiasAdd_1_grad/BiasAddGrad0^gradients/rnn/while/gru_cell/Tanh_grad/TanhGrad
╖
Dgradients/rnn/while/gru_cell/BiasAdd_1_grad/tuple/control_dependencyIdentity/gradients/rnn/while/gru_cell/Tanh_grad/TanhGrad=^gradients/rnn/while/gru_cell/BiasAdd_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn/while/gru_cell/Tanh_grad/TanhGrad*(
_output_shapes
:         О
╝
Fgradients/rnn/while/gru_cell/BiasAdd_1_grad/tuple/control_dependency_1Identity7gradients/rnn/while/gru_cell/BiasAdd_1_grad/BiasAddGrad=^gradients/rnn/while/gru_cell/BiasAdd_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/rnn/while/gru_cell/BiasAdd_1_grad/BiasAddGrad*
_output_shapes	
:О
У
1gradients/rnn/while/gru_cell/MatMul_1_grad/MatMulMatMulDgradients/rnn/while/gru_cell/BiasAdd_1_grad/tuple/control_dependency7gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul/Enter*
transpose_b(*
T0*(
_output_shapes
:         Э*
transpose_a( 
Ё
7gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul/EnterEnter"rnn/gru_cell/candidate/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
ЭО*1

frame_name#!gradients/rnn/while/while_context
Ф
3gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1MatMul>gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/StackPopV2Dgradients/rnn/while/gru_cell/BiasAdd_1_grad/tuple/control_dependency* 
_output_shapes
:
ЭО*
transpose_a(*
transpose_b( *
T0
┤
9gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/ConstConst*.
_class$
" loc:@rnn/while/gru_cell/concat_1*
valueB :
         *
dtype0*
_output_shapes
: 
ю
9gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/f_accStackV29gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/Const*
	elem_type0*.
_class$
" loc:@rnn/while/gru_cell/concat_1*

stack_name *
_output_shapes
:
∙
9gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/EnterEnter9gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context
№
?gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/StackPushV2StackPushV29gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/Enterrnn/while/gru_cell/concat_1^gradients/Add*
T0*(
_output_shapes
:         Э*
swap_memory( 
▌
>gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/StackPopV2
StackPopV2Dgradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*(
_output_shapes
:         Э*
	elem_type0
О
Dgradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/StackPopV2/EnterEnter9gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context
н
;gradients/rnn/while/gru_cell/MatMul_1_grad/tuple/group_depsNoOp2^gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul4^gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1
╣
Cgradients/rnn/while/gru_cell/MatMul_1_grad/tuple/control_dependencyIdentity1gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul<^gradients/rnn/while/gru_cell/MatMul_1_grad/tuple/group_deps*(
_output_shapes
:         Э*
T0*D
_class:
86loc:@gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul
╖
Egradients/rnn/while/gru_cell/MatMul_1_grad/tuple/control_dependency_1Identity3gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1<^gradients/rnn/while/gru_cell/MatMul_1_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1* 
_output_shapes
:
ЭО
Ж
7gradients/rnn/while/gru_cell/BiasAdd_1/Enter_grad/b_accConst*
dtype0*
_output_shapes	
:О*
valueBО*    
В
9gradients/rnn/while/gru_cell/BiasAdd_1/Enter_grad/b_acc_1Enter7gradients/rnn/while/gru_cell/BiasAdd_1/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes	
:О*1

frame_name#!gradients/rnn/while/while_context
я
9gradients/rnn/while/gru_cell/BiasAdd_1/Enter_grad/b_acc_2Merge9gradients/rnn/while/gru_cell/BiasAdd_1/Enter_grad/b_acc_1?gradients/rnn/while/gru_cell/BiasAdd_1/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:О: 
┐
8gradients/rnn/while/gru_cell/BiasAdd_1/Enter_grad/SwitchSwitch9gradients/rnn/while/gru_cell/BiasAdd_1/Enter_grad/b_acc_2gradients/b_count_2*
T0*"
_output_shapes
:О:О
ц
5gradients/rnn/while/gru_cell/BiasAdd_1/Enter_grad/AddAdd:gradients/rnn/while/gru_cell/BiasAdd_1/Enter_grad/Switch:1Fgradients/rnn/while/gru_cell/BiasAdd_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:О
н
?gradients/rnn/while/gru_cell/BiasAdd_1/Enter_grad/NextIterationNextIteration5gradients/rnn/while/gru_cell/BiasAdd_1/Enter_grad/Add*
T0*
_output_shapes	
:О
б
9gradients/rnn/while/gru_cell/BiasAdd_1/Enter_grad/b_acc_3Exit8gradients/rnn/while/gru_cell/BiasAdd_1/Enter_grad/Switch*
_output_shapes	
:О*
T0
В
0gradients/rnn/while/gru_cell/concat_1_grad/ConstConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
Б
/gradients/rnn/while/gru_cell/concat_1_grad/RankConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
╛
.gradients/rnn/while/gru_cell/concat_1_grad/modFloorMod0gradients/rnn/while/gru_cell/concat_1_grad/Const/gradients/rnn/while/gru_cell/concat_1_grad/Rank*
T0*
_output_shapes
: 
Л
0gradients/rnn/while/gru_cell/concat_1_grad/ShapeShapernn/while/TensorArrayReadV3*
_output_shapes
:*
T0*
out_type0
¤
1gradients/rnn/while/gru_cell/concat_1_grad/ShapeNShapeN<gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPopV2>gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPopV2_1*
N* 
_output_shapes
::*
T0*
out_type0
▓
7gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/ConstConst*.
_class$
" loc:@rnn/while/TensorArrayReadV3*
valueB :
         *
dtype0*
_output_shapes
: 
ъ
7gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/f_accStackV27gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/Const*.
_class$
" loc:@rnn/while/TensorArrayReadV3*

stack_name *
_output_shapes
:*
	elem_type0
ї
7gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/EnterEnter7gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context
ў
=gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPushV2StackPushV27gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/Enterrnn/while/TensorArrayReadV3^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
╪
<gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPopV2
StackPopV2Bgradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
К
Bgradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPopV2/EnterEnter7gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context
п
9gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/Const_1Const*)
_class
loc:@rnn/while/gru_cell/mul*
valueB :
         *
dtype0*
_output_shapes
: 
щ
9gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/f_acc_1StackV29gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/Const_1*
	elem_type0*)
_class
loc:@rnn/while/gru_cell/mul*

stack_name *
_output_shapes
:
∙
9gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/Enter_1Enter9gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context
ў
?gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPushV2_1StackPushV29gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/Enter_1rnn/while/gru_cell/mul^gradients/Add*(
_output_shapes
:         О*
swap_memory( *
T0
▌
>gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPopV2_1
StackPopV2Dgradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPopV2_1/Enter^gradients/Sub*(
_output_shapes
:         О*
	elem_type0
О
Dgradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPopV2_1/EnterEnter9gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context
К
7gradients/rnn/while/gru_cell/concat_1_grad/ConcatOffsetConcatOffset.gradients/rnn/while/gru_cell/concat_1_grad/mod1gradients/rnn/while/gru_cell/concat_1_grad/ShapeN3gradients/rnn/while/gru_cell/concat_1_grad/ShapeN:1*
N* 
_output_shapes
::
▓
0gradients/rnn/while/gru_cell/concat_1_grad/SliceSliceCgradients/rnn/while/gru_cell/MatMul_1_grad/tuple/control_dependency7gradients/rnn/while/gru_cell/concat_1_grad/ConcatOffset1gradients/rnn/while/gru_cell/concat_1_grad/ShapeN*
Index0*
T0*0
_output_shapes
:                  
╕
2gradients/rnn/while/gru_cell/concat_1_grad/Slice_1SliceCgradients/rnn/while/gru_cell/MatMul_1_grad/tuple/control_dependency9gradients/rnn/while/gru_cell/concat_1_grad/ConcatOffset:13gradients/rnn/while/gru_cell/concat_1_grad/ShapeN:1*
Index0*
T0*0
_output_shapes
:                  
л
;gradients/rnn/while/gru_cell/concat_1_grad/tuple/group_depsNoOp1^gradients/rnn/while/gru_cell/concat_1_grad/Slice3^gradients/rnn/while/gru_cell/concat_1_grad/Slice_1
╢
Cgradients/rnn/while/gru_cell/concat_1_grad/tuple/control_dependencyIdentity0gradients/rnn/while/gru_cell/concat_1_grad/Slice<^gradients/rnn/while/gru_cell/concat_1_grad/tuple/group_deps*'
_output_shapes
:         *
T0*C
_class9
75loc:@gradients/rnn/while/gru_cell/concat_1_grad/Slice
╜
Egradients/rnn/while/gru_cell/concat_1_grad/tuple/control_dependency_1Identity2gradients/rnn/while/gru_cell/concat_1_grad/Slice_1<^gradients/rnn/while/gru_cell/concat_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/while/gru_cell/concat_1_grad/Slice_1*(
_output_shapes
:         О
П
6gradients/rnn/while/gru_cell/MatMul_1/Enter_grad/b_accConst*
valueB
ЭО*    *
dtype0* 
_output_shapes
:
ЭО
Е
8gradients/rnn/while/gru_cell/MatMul_1/Enter_grad/b_acc_1Enter6gradients/rnn/while/gru_cell/MatMul_1/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations * 
_output_shapes
:
ЭО*1

frame_name#!gradients/rnn/while/while_context
ё
8gradients/rnn/while/gru_cell/MatMul_1/Enter_grad/b_acc_2Merge8gradients/rnn/while/gru_cell/MatMul_1/Enter_grad/b_acc_1>gradients/rnn/while/gru_cell/MatMul_1/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
ЭО: 
╟
7gradients/rnn/while/gru_cell/MatMul_1/Enter_grad/SwitchSwitch8gradients/rnn/while/gru_cell/MatMul_1/Enter_grad/b_acc_2gradients/b_count_2*
T0*,
_output_shapes
:
ЭО:
ЭО
ш
4gradients/rnn/while/gru_cell/MatMul_1/Enter_grad/AddAdd9gradients/rnn/while/gru_cell/MatMul_1/Enter_grad/Switch:1Egradients/rnn/while/gru_cell/MatMul_1_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
ЭО
░
>gradients/rnn/while/gru_cell/MatMul_1/Enter_grad/NextIterationNextIteration4gradients/rnn/while/gru_cell/MatMul_1/Enter_grad/Add*
T0* 
_output_shapes
:
ЭО
д
8gradients/rnn/while/gru_cell/MatMul_1/Enter_grad/b_acc_3Exit7gradients/rnn/while/gru_cell/MatMul_1/Enter_grad/Switch*
T0* 
_output_shapes
:
ЭО
Г
+gradients/rnn/while/gru_cell/mul_grad/ShapeShapernn/while/gru_cell/split*
T0*
out_type0*
_output_shapes
:
Б
-gradients/rnn/while/gru_cell/mul_grad/Shape_1Shapernn/while/Identity_3*
T0*
out_type0*
_output_shapes
:
г
;gradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPopV2Hgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
╠
Agradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/ConstConst*>
_class4
20loc:@gradients/rnn/while/gru_cell/mul_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
О
Agradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/f_accStackV2Agradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*>
_class4
20loc:@gradients/rnn/while/gru_cell/mul_grad/Shape
Й
Agradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/EnterEnterAgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context
О
Ggradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/Enter+gradients/rnn/while/gru_cell/mul_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
▀
Fgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Lgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
Ю
Lgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterAgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context
╨
Cgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/Const_1Const*@
_class6
42loc:@gradients/rnn/while/gru_cell/mul_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
Ф
Cgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2Cgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/Const_1*@
_class6
42loc:@gradients/rnn/while/gru_cell/mul_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
Н
Cgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/Enter_1EnterCgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context*
T0*
is_constant(
Ф
Igradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Cgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/Enter_1-gradients/rnn/while/gru_cell/mul_grad/Shape_1^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
у
Hgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Ngradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
в
Ngradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterCgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context
т
)gradients/rnn/while/gru_cell/mul_grad/MulMulEgradients/rnn/while/gru_cell/concat_1_grad/tuple/control_dependency_16gradients/rnn/while/gru_cell/mul_1_grad/Mul/StackPopV2*(
_output_shapes
:         О*
T0
╪
)gradients/rnn/while/gru_cell/mul_grad/SumSum)gradients/rnn/while/gru_cell/mul_grad/Mul;gradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ь
-gradients/rnn/while/gru_cell/mul_grad/ReshapeReshape)gradients/rnn/while/gru_cell/mul_grad/SumFgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPopV2*(
_output_shapes
:         О*
T0*
Tshape0
ф
+gradients/rnn/while/gru_cell/mul_grad/Mul_1Mul6gradients/rnn/while/gru_cell/mul_grad/Mul_1/StackPopV2Egradients/rnn/while/gru_cell/concat_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         О
й
1gradients/rnn/while/gru_cell/mul_grad/Mul_1/ConstConst*+
_class!
loc:@rnn/while/gru_cell/split*
valueB :
         *
dtype0*
_output_shapes
: 
█
1gradients/rnn/while/gru_cell/mul_grad/Mul_1/f_accStackV21gradients/rnn/while/gru_cell/mul_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*+
_class!
loc:@rnn/while/gru_cell/split
щ
1gradients/rnn/while/gru_cell/mul_grad/Mul_1/EnterEnter1gradients/rnn/while/gru_cell/mul_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context*
T0*
is_constant(
щ
7gradients/rnn/while/gru_cell/mul_grad/Mul_1/StackPushV2StackPushV21gradients/rnn/while/gru_cell/mul_grad/Mul_1/Enterrnn/while/gru_cell/split^gradients/Add*
T0*(
_output_shapes
:         О*
swap_memory( 
═
6gradients/rnn/while/gru_cell/mul_grad/Mul_1/StackPopV2
StackPopV2<gradients/rnn/while/gru_cell/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*(
_output_shapes
:         О
■
<gradients/rnn/while/gru_cell/mul_grad/Mul_1/StackPopV2/EnterEnter1gradients/rnn/while/gru_cell/mul_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0*
is_constant(
▐
+gradients/rnn/while/gru_cell/mul_grad/Sum_1Sum+gradients/rnn/while/gru_cell/mul_grad/Mul_1=gradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Є
/gradients/rnn/while/gru_cell/mul_grad/Reshape_1Reshape+gradients/rnn/while/gru_cell/mul_grad/Sum_1Hgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         О
а
6gradients/rnn/while/gru_cell/mul_grad/tuple/group_depsNoOp.^gradients/rnn/while/gru_cell/mul_grad/Reshape0^gradients/rnn/while/gru_cell/mul_grad/Reshape_1
з
>gradients/rnn/while/gru_cell/mul_grad/tuple/control_dependencyIdentity-gradients/rnn/while/gru_cell/mul_grad/Reshape7^gradients/rnn/while/gru_cell/mul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn/while/gru_cell/mul_grad/Reshape*(
_output_shapes
:         О
н
@gradients/rnn/while/gru_cell/mul_grad/tuple/control_dependency_1Identity/gradients/rnn/while/gru_cell/mul_grad/Reshape_17^gradients/rnn/while/gru_cell/mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn/while/gru_cell/mul_grad/Reshape_1*(
_output_shapes
:         О
Ь
gradients/AddN_1AddN@gradients/rnn/while/gru_cell/mul_1_grad/tuple/control_dependency@gradients/rnn/while/gru_cell/sub_grad/tuple/control_dependency_1*
T0*B
_class8
64loc:@gradients/rnn/while/gru_cell/mul_1_grad/Reshape*
N*(
_output_shapes
:         О
К
.gradients/rnn/while/gru_cell/split_grad/concatConcatV2>gradients/rnn/while/gru_cell/mul_grad/tuple/control_dependencygradients/AddN_14gradients/rnn/while/gru_cell/split_grad/concat/Const*

Tidx0*
T0*
N*(
_output_shapes
:         Ь
Ж
4gradients/rnn/while/gru_cell/split_grad/concat/ConstConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
щ
5gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGradSigmoidGrad@gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/StackPopV2.gradients/rnn/while/gru_cell/split_grad/concat*
T0*(
_output_shapes
:         Ь
╡
;gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/ConstConst*-
_class#
!loc:@rnn/while/gru_cell/Sigmoid*
valueB :
         *
dtype0*
_output_shapes
: 
ё
;gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/f_accStackV2;gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/Const*-
_class#
!loc:@rnn/while/gru_cell/Sigmoid*

stack_name *
_output_shapes
:*
	elem_type0
¤
;gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/EnterEnter;gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context
 
Agradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/StackPushV2StackPushV2;gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/Enterrnn/while/gru_cell/Sigmoid^gradients/Add*
T0*(
_output_shapes
:         Ь*
swap_memory( 
с
@gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/StackPopV2
StackPopV2Fgradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/StackPopV2/Enter^gradients/Sub*
	elem_type0*(
_output_shapes
:         Ь
Т
Fgradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/StackPopV2/EnterEnter;gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context
╕
5gradients/rnn/while/gru_cell/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes	
:Ь*
T0
▓
:gradients/rnn/while/gru_cell/BiasAdd_grad/tuple/group_depsNoOp6^gradients/rnn/while/gru_cell/BiasAdd_grad/BiasAddGrad6^gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad
┐
Bgradients/rnn/while/gru_cell/BiasAdd_grad/tuple/control_dependencyIdentity5gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad;^gradients/rnn/while/gru_cell/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad*(
_output_shapes
:         Ь
┤
Dgradients/rnn/while/gru_cell/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/rnn/while/gru_cell/BiasAdd_grad/BiasAddGrad;^gradients/rnn/while/gru_cell/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn/while/gru_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:Ь
Н
/gradients/rnn/while/gru_cell/MatMul_grad/MatMulMatMulBgradients/rnn/while/gru_cell/BiasAdd_grad/tuple/control_dependency5gradients/rnn/while/gru_cell/MatMul_grad/MatMul/Enter*
transpose_b(*
T0*(
_output_shapes
:         Э*
transpose_a( 
ъ
5gradients/rnn/while/gru_cell/MatMul_grad/MatMul/EnterEnterrnn/gru_cell/gates/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
ЭЬ*1

frame_name#!gradients/rnn/while/while_context
О
1gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1MatMul<gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/StackPopV2Bgradients/rnn/while/gru_cell/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
ЭЬ*
transpose_a(*
transpose_b( 
░
7gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/ConstConst*,
_class"
 loc:@rnn/while/gru_cell/concat*
valueB :
         *
dtype0*
_output_shapes
: 
ш
7gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/f_accStackV27gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/Const*,
_class"
 loc:@rnn/while/gru_cell/concat*

stack_name *
_output_shapes
:*
	elem_type0
ї
7gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/EnterEnter7gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*'

frame_namernn/while/while_context
Ў
=gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV27gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/Enterrnn/while/gru_cell/concat^gradients/Add*(
_output_shapes
:         Э*
swap_memory( *
T0
┘
<gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Bgradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*(
_output_shapes
:         Э*
	elem_type0
К
Bgradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnter7gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context
з
9gradients/rnn/while/gru_cell/MatMul_grad/tuple/group_depsNoOp0^gradients/rnn/while/gru_cell/MatMul_grad/MatMul2^gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1
▒
Agradients/rnn/while/gru_cell/MatMul_grad/tuple/control_dependencyIdentity/gradients/rnn/while/gru_cell/MatMul_grad/MatMul:^gradients/rnn/while/gru_cell/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn/while/gru_cell/MatMul_grad/MatMul*(
_output_shapes
:         Э
п
Cgradients/rnn/while/gru_cell/MatMul_grad/tuple/control_dependency_1Identity1gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1:^gradients/rnn/while/gru_cell/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
ЭЬ
Д
5gradients/rnn/while/gru_cell/BiasAdd/Enter_grad/b_accConst*
valueBЬ*    *
dtype0*
_output_shapes	
:Ь
■
7gradients/rnn/while/gru_cell/BiasAdd/Enter_grad/b_acc_1Enter5gradients/rnn/while/gru_cell/BiasAdd/Enter_grad/b_acc*
parallel_iterations *
_output_shapes	
:Ь*1

frame_name#!gradients/rnn/while/while_context*
T0*
is_constant( 
щ
7gradients/rnn/while/gru_cell/BiasAdd/Enter_grad/b_acc_2Merge7gradients/rnn/while/gru_cell/BiasAdd/Enter_grad/b_acc_1=gradients/rnn/while/gru_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:Ь: 
╗
6gradients/rnn/while/gru_cell/BiasAdd/Enter_grad/SwitchSwitch7gradients/rnn/while/gru_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0*"
_output_shapes
:Ь:Ь
р
3gradients/rnn/while/gru_cell/BiasAdd/Enter_grad/AddAdd8gradients/rnn/while/gru_cell/BiasAdd/Enter_grad/Switch:1Dgradients/rnn/while/gru_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:Ь
й
=gradients/rnn/while/gru_cell/BiasAdd/Enter_grad/NextIterationNextIteration3gradients/rnn/while/gru_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:Ь
Э
7gradients/rnn/while/gru_cell/BiasAdd/Enter_grad/b_acc_3Exit6gradients/rnn/while/gru_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:Ь*
T0
А
.gradients/rnn/while/gru_cell/concat_grad/ConstConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

-gradients/rnn/while/gru_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
╕
,gradients/rnn/while/gru_cell/concat_grad/modFloorMod.gradients/rnn/while/gru_cell/concat_grad/Const-gradients/rnn/while/gru_cell/concat_grad/Rank*
_output_shapes
: *
T0
Й
.gradients/rnn/while/gru_cell/concat_grad/ShapeShapernn/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
є
/gradients/rnn/while/gru_cell/concat_grad/ShapeNShapeN<gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPopV26gradients/rnn/while/gru_cell/mul_1_grad/Mul/StackPopV2*
N* 
_output_shapes
::*
T0*
out_type0
В
5gradients/rnn/while/gru_cell/concat_grad/ConcatOffsetConcatOffset,gradients/rnn/while/gru_cell/concat_grad/mod/gradients/rnn/while/gru_cell/concat_grad/ShapeN1gradients/rnn/while/gru_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
к
.gradients/rnn/while/gru_cell/concat_grad/SliceSliceAgradients/rnn/while/gru_cell/MatMul_grad/tuple/control_dependency5gradients/rnn/while/gru_cell/concat_grad/ConcatOffset/gradients/rnn/while/gru_cell/concat_grad/ShapeN*0
_output_shapes
:                  *
Index0*
T0
░
0gradients/rnn/while/gru_cell/concat_grad/Slice_1SliceAgradients/rnn/while/gru_cell/MatMul_grad/tuple/control_dependency7gradients/rnn/while/gru_cell/concat_grad/ConcatOffset:11gradients/rnn/while/gru_cell/concat_grad/ShapeN:1*
Index0*
T0*0
_output_shapes
:                  
е
9gradients/rnn/while/gru_cell/concat_grad/tuple/group_depsNoOp/^gradients/rnn/while/gru_cell/concat_grad/Slice1^gradients/rnn/while/gru_cell/concat_grad/Slice_1
о
Agradients/rnn/while/gru_cell/concat_grad/tuple/control_dependencyIdentity.gradients/rnn/while/gru_cell/concat_grad/Slice:^gradients/rnn/while/gru_cell/concat_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn/while/gru_cell/concat_grad/Slice*'
_output_shapes
:         
╡
Cgradients/rnn/while/gru_cell/concat_grad/tuple/control_dependency_1Identity0gradients/rnn/while/gru_cell/concat_grad/Slice_1:^gradients/rnn/while/gru_cell/concat_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/while/gru_cell/concat_grad/Slice_1*(
_output_shapes
:         О
Н
4gradients/rnn/while/gru_cell/MatMul/Enter_grad/b_accConst*
dtype0* 
_output_shapes
:
ЭЬ*
valueB
ЭЬ*    
Б
6gradients/rnn/while/gru_cell/MatMul/Enter_grad/b_acc_1Enter4gradients/rnn/while/gru_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations * 
_output_shapes
:
ЭЬ*1

frame_name#!gradients/rnn/while/while_context
ы
6gradients/rnn/while/gru_cell/MatMul/Enter_grad/b_acc_2Merge6gradients/rnn/while/gru_cell/MatMul/Enter_grad/b_acc_1<gradients/rnn/while/gru_cell/MatMul/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
ЭЬ: 
├
5gradients/rnn/while/gru_cell/MatMul/Enter_grad/SwitchSwitch6gradients/rnn/while/gru_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_output_shapes
:
ЭЬ:
ЭЬ*
T0
т
2gradients/rnn/while/gru_cell/MatMul/Enter_grad/AddAdd7gradients/rnn/while/gru_cell/MatMul/Enter_grad/Switch:1Cgradients/rnn/while/gru_cell/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
ЭЬ
м
<gradients/rnn/while/gru_cell/MatMul/Enter_grad/NextIterationNextIteration2gradients/rnn/while/gru_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:
ЭЬ
а
6gradients/rnn/while/gru_cell/MatMul/Enter_grad/b_acc_3Exit5gradients/rnn/while/gru_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:
ЭЬ
х
gradients/AddN_2AddNBgradients/rnn/while/gru_cell/mul_1_grad/tuple/control_dependency_1@gradients/rnn/while/gru_cell/mul_grad/tuple/control_dependency_1Cgradients/rnn/while/gru_cell/concat_grad/tuple/control_dependency_1*
T0*D
_class:
86loc:@gradients/rnn/while/gru_cell/mul_1_grad/Reshape_1*
N*(
_output_shapes
:         О
З
1gradients/rnn/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_2*(
_output_shapes
:         О*
T0
}
beta1_power/initial_valueConst*
_class
loc:@dense/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
О
beta1_power
VariableV2*
shared_name *
_class
loc:@dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
н
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
i
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*
_class
loc:@dense/bias
}
beta2_power/initial_valueConst*
_class
loc:@dense/bias*
valueB
 *w╛?*
dtype0*
_output_shapes
: 
О
beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@dense/bias
н
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@dense/bias
i
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
┐
@rnn/gru_cell/gates/kernel/Adam/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
valueB"    *
dtype0*
_output_shapes
:
й
6rnn/gru_cell/gates/kernel/Adam/Initializer/zeros/ConstConst*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Э
0rnn/gru_cell/gates/kernel/Adam/Initializer/zerosFill@rnn/gru_cell/gates/kernel/Adam/Initializer/zeros/shape_as_tensor6rnn/gru_cell/gates/kernel/Adam/Initializer/zeros/Const*
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel*

index_type0* 
_output_shapes
:
ЭЬ
─
rnn/gru_cell/gates/kernel/Adam
VariableV2*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
	container *
shape:
ЭЬ*
dtype0* 
_output_shapes
:
ЭЬ*
shared_name 
Г
%rnn/gru_cell/gates/kernel/Adam/AssignAssignrnn/gru_cell/gates/kernel/Adam0rnn/gru_cell/gates/kernel/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
validate_shape(* 
_output_shapes
:
ЭЬ
и
#rnn/gru_cell/gates/kernel/Adam/readIdentityrnn/gru_cell/gates/kernel/Adam*
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel* 
_output_shapes
:
ЭЬ
┴
Brnn/gru_cell/gates/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
valueB"    
л
8rnn/gru_cell/gates/kernel/Adam_1/Initializer/zeros/ConstConst*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
г
2rnn/gru_cell/gates/kernel/Adam_1/Initializer/zerosFillBrnn/gru_cell/gates/kernel/Adam_1/Initializer/zeros/shape_as_tensor8rnn/gru_cell/gates/kernel/Adam_1/Initializer/zeros/Const*
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel*

index_type0* 
_output_shapes
:
ЭЬ
╞
 rnn/gru_cell/gates/kernel/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@rnn/gru_cell/gates/kernel*
	container *
shape:
ЭЬ*
dtype0* 
_output_shapes
:
ЭЬ
Й
'rnn/gru_cell/gates/kernel/Adam_1/AssignAssign rnn/gru_cell/gates/kernel/Adam_12rnn/gru_cell/gates/kernel/Adam_1/Initializer/zeros*
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
validate_shape(* 
_output_shapes
:
ЭЬ*
use_locking(
м
%rnn/gru_cell/gates/kernel/Adam_1/readIdentity rnn/gru_cell/gates/kernel/Adam_1*
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel* 
_output_shapes
:
ЭЬ
й
.rnn/gru_cell/gates/bias/Adam/Initializer/zerosConst**
_class 
loc:@rnn/gru_cell/gates/bias*
valueBЬ*    *
dtype0*
_output_shapes	
:Ь
╢
rnn/gru_cell/gates/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:Ь*
shared_name **
_class 
loc:@rnn/gru_cell/gates/bias*
	container *
shape:Ь
Ў
#rnn/gru_cell/gates/bias/Adam/AssignAssignrnn/gru_cell/gates/bias/Adam.rnn/gru_cell/gates/bias/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@rnn/gru_cell/gates/bias*
validate_shape(*
_output_shapes	
:Ь
Э
!rnn/gru_cell/gates/bias/Adam/readIdentityrnn/gru_cell/gates/bias/Adam*
T0**
_class 
loc:@rnn/gru_cell/gates/bias*
_output_shapes	
:Ь
л
0rnn/gru_cell/gates/bias/Adam_1/Initializer/zerosConst**
_class 
loc:@rnn/gru_cell/gates/bias*
valueBЬ*    *
dtype0*
_output_shapes	
:Ь
╕
rnn/gru_cell/gates/bias/Adam_1
VariableV2*
shared_name **
_class 
loc:@rnn/gru_cell/gates/bias*
	container *
shape:Ь*
dtype0*
_output_shapes	
:Ь
№
%rnn/gru_cell/gates/bias/Adam_1/AssignAssignrnn/gru_cell/gates/bias/Adam_10rnn/gru_cell/gates/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:Ь*
use_locking(*
T0**
_class 
loc:@rnn/gru_cell/gates/bias
б
#rnn/gru_cell/gates/bias/Adam_1/readIdentityrnn/gru_cell/gates/bias/Adam_1*
T0**
_class 
loc:@rnn/gru_cell/gates/bias*
_output_shapes	
:Ь
╟
Drnn/gru_cell/candidate/kernel/Adam/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
valueB"    *
dtype0*
_output_shapes
:
▒
:rnn/gru_cell/candidate/kernel/Adam/Initializer/zeros/ConstConst*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
н
4rnn/gru_cell/candidate/kernel/Adam/Initializer/zerosFillDrnn/gru_cell/candidate/kernel/Adam/Initializer/zeros/shape_as_tensor:rnn/gru_cell/candidate/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
ЭО*
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*

index_type0
╠
"rnn/gru_cell/candidate/kernel/Adam
VariableV2*
shape:
ЭО*
dtype0* 
_output_shapes
:
ЭО*
shared_name *0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
	container 
У
)rnn/gru_cell/candidate/kernel/Adam/AssignAssign"rnn/gru_cell/candidate/kernel/Adam4rnn/gru_cell/candidate/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
ЭО*
use_locking(*
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel
┤
'rnn/gru_cell/candidate/kernel/Adam/readIdentity"rnn/gru_cell/candidate/kernel/Adam*
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel* 
_output_shapes
:
ЭО
╔
Frnn/gru_cell/candidate/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
valueB"    
│
<rnn/gru_cell/candidate/kernel/Adam_1/Initializer/zeros/ConstConst*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
│
6rnn/gru_cell/candidate/kernel/Adam_1/Initializer/zerosFillFrnn/gru_cell/candidate/kernel/Adam_1/Initializer/zeros/shape_as_tensor<rnn/gru_cell/candidate/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
ЭО*
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*

index_type0
╬
$rnn/gru_cell/candidate/kernel/Adam_1
VariableV2*
shared_name *0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
	container *
shape:
ЭО*
dtype0* 
_output_shapes
:
ЭО
Щ
+rnn/gru_cell/candidate/kernel/Adam_1/AssignAssign$rnn/gru_cell/candidate/kernel/Adam_16rnn/gru_cell/candidate/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
ЭО*
use_locking(*
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel
╕
)rnn/gru_cell/candidate/kernel/Adam_1/readIdentity$rnn/gru_cell/candidate/kernel/Adam_1*
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel* 
_output_shapes
:
ЭО
▒
2rnn/gru_cell/candidate/bias/Adam/Initializer/zerosConst*.
_class$
" loc:@rnn/gru_cell/candidate/bias*
valueBО*    *
dtype0*
_output_shapes	
:О
╛
 rnn/gru_cell/candidate/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:О*
shared_name *.
_class$
" loc:@rnn/gru_cell/candidate/bias*
	container *
shape:О
Ж
'rnn/gru_cell/candidate/bias/Adam/AssignAssign rnn/gru_cell/candidate/bias/Adam2rnn/gru_cell/candidate/bias/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@rnn/gru_cell/candidate/bias*
validate_shape(*
_output_shapes	
:О
й
%rnn/gru_cell/candidate/bias/Adam/readIdentity rnn/gru_cell/candidate/bias/Adam*
T0*.
_class$
" loc:@rnn/gru_cell/candidate/bias*
_output_shapes	
:О
│
4rnn/gru_cell/candidate/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:О*.
_class$
" loc:@rnn/gru_cell/candidate/bias*
valueBО*    
└
"rnn/gru_cell/candidate/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:О*
shared_name *.
_class$
" loc:@rnn/gru_cell/candidate/bias*
	container *
shape:О
М
)rnn/gru_cell/candidate/bias/Adam_1/AssignAssign"rnn/gru_cell/candidate/bias/Adam_14rnn/gru_cell/candidate/bias/Adam_1/Initializer/zeros*
T0*.
_class$
" loc:@rnn/gru_cell/candidate/bias*
validate_shape(*
_output_shapes	
:О*
use_locking(
н
'rnn/gru_cell/candidate/bias/Adam_1/readIdentity"rnn/gru_cell/candidate/bias/Adam_1*
T0*.
_class$
" loc:@rnn/gru_cell/candidate/bias*
_output_shapes	
:О
Ы
#dense/kernel/Adam/Initializer/zerosConst*
_class
loc:@dense/kernel*
valueB	О*    *
dtype0*
_output_shapes
:	О
и
dense/kernel/Adam
VariableV2*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:	О*
dtype0*
_output_shapes
:	О
╬
dense/kernel/Adam/AssignAssigndense/kernel/Adam#dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	О
А
dense/kernel/Adam/readIdentitydense/kernel/Adam*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	О
Э
%dense/kernel/Adam_1/Initializer/zerosConst*
_class
loc:@dense/kernel*
valueB	О*    *
dtype0*
_output_shapes
:	О
к
dense/kernel/Adam_1
VariableV2*
	container *
shape:	О*
dtype0*
_output_shapes
:	О*
shared_name *
_class
loc:@dense/kernel
╘
dense/kernel/Adam_1/AssignAssigndense/kernel/Adam_1%dense/kernel/Adam_1/Initializer/zeros*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	О*
use_locking(
Д
dense/kernel/Adam_1/readIdentitydense/kernel/Adam_1*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	О
Н
!dense/bias/Adam/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB*    *
dtype0*
_output_shapes
:
Ъ
dense/bias/Adam
VariableV2*
shared_name *
_class
loc:@dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
┴
dense/bias/Adam/AssignAssigndense/bias/Adam!dense/bias/Adam/Initializer/zeros*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(
u
dense/bias/Adam/readIdentitydense/bias/Adam*
_output_shapes
:*
T0*
_class
loc:@dense/bias
П
#dense/bias/Adam_1/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB*    *
dtype0*
_output_shapes
:
Ь
dense/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@dense/bias
╟
dense/bias/Adam_1/AssignAssigndense/bias/Adam_1#dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense/bias
y
dense/bias/Adam_1/readIdentitydense/bias/Adam_1*
_output_shapes
:*
T0*
_class
loc:@dense/bias
W
Adam/learning_rateConst*
valueB
 *
╫#<*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w╛?
Q
Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
п
/Adam/update_rnn/gru_cell/gates/kernel/ApplyAdam	ApplyAdamrnn/gru_cell/gates/kernelrnn/gru_cell/gates/kernel/Adam rnn/gru_cell/gates/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/rnn/while/gru_cell/MatMul/Enter_grad/b_acc_3*
use_nesterov( * 
_output_shapes
:
ЭЬ*
use_locking( *
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel
б
-Adam/update_rnn/gru_cell/gates/bias/ApplyAdam	ApplyAdamrnn/gru_cell/gates/biasrnn/gru_cell/gates/bias/Adamrnn/gru_cell/gates/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/rnn/while/gru_cell/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0**
_class 
loc:@rnn/gru_cell/gates/bias*
use_nesterov( *
_output_shapes	
:Ь
┼
3Adam/update_rnn/gru_cell/candidate/kernel/ApplyAdam	ApplyAdamrnn/gru_cell/candidate/kernel"rnn/gru_cell/candidate/kernel/Adam$rnn/gru_cell/candidate/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon8gradients/rnn/while/gru_cell/MatMul_1/Enter_grad/b_acc_3*
use_nesterov( * 
_output_shapes
:
ЭО*
use_locking( *
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel
╖
1Adam/update_rnn/gru_cell/candidate/bias/ApplyAdam	ApplyAdamrnn/gru_cell/candidate/bias rnn/gru_cell/candidate/bias/Adam"rnn/gru_cell/candidate/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon9gradients/rnn/while/gru_cell/BiasAdd_1/Enter_grad/b_acc_3*
T0*.
_class$
" loc:@rnn/gru_cell/candidate/bias*
use_nesterov( *
_output_shapes	
:О*
use_locking( 
э
"Adam/update_dense/kernel/ApplyAdam	ApplyAdamdense/kerneldense/kernel/Adamdense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/kernel*
use_nesterov( *
_output_shapes
:	О
▀
 Adam/update_dense/bias/ApplyAdam	ApplyAdam
dense/biasdense/bias/Adamdense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/bias*
use_nesterov( *
_output_shapes
:
Б
Adam/mulMulbeta1_power/read
Adam/beta1!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam2^Adam/update_rnn/gru_cell/candidate/bias/ApplyAdam4^Adam/update_rnn/gru_cell/candidate/kernel/ApplyAdam.^Adam/update_rnn/gru_cell/gates/bias/ApplyAdam0^Adam/update_rnn/gru_cell/gates/kernel/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@dense/bias
Х
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@dense/bias
Г

Adam/mul_1Mulbeta2_power/read
Adam/beta2!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam2^Adam/update_rnn/gru_cell/candidate/bias/ApplyAdam4^Adam/update_rnn/gru_cell/candidate/kernel/ApplyAdam.^Adam/update_rnn/gru_cell/gates/bias/ApplyAdam0^Adam/update_rnn/gru_cell/gates/kernel/ApplyAdam*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
Щ
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
╛
AdamNoOp^Adam/Assign^Adam/Assign_1!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam2^Adam/update_rnn/gru_cell/candidate/bias/ApplyAdam4^Adam/update_rnn/gru_cell/candidate/kernel/ApplyAdam.^Adam/update_rnn/gru_cell/gates/bias/ApplyAdam0^Adam/update_rnn/gru_cell/gates/kernel/ApplyAdam
N
	loss/tagsConst*
dtype0*
_output_shapes
: *
valueB
 Bloss
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 
V
accuracy/tagsConst*
valueB Baccuracy*
dtype0*
_output_shapes
: 
Q
accuracyScalarSummaryaccuracy/tagsMean_1*
T0*
_output_shapes
: 
S
Merge/MergeSummaryMergeSummarylossaccuracy*
N*
_output_shapes
: 
м
initNoOp^beta1_power/Assign^beta2_power/Assign^dense/bias/Adam/Assign^dense/bias/Adam_1/Assign^dense/bias/Assign^dense/kernel/Adam/Assign^dense/kernel/Adam_1/Assign^dense/kernel/Assign(^rnn/gru_cell/candidate/bias/Adam/Assign*^rnn/gru_cell/candidate/bias/Adam_1/Assign#^rnn/gru_cell/candidate/bias/Assign*^rnn/gru_cell/candidate/kernel/Adam/Assign,^rnn/gru_cell/candidate/kernel/Adam_1/Assign%^rnn/gru_cell/candidate/kernel/Assign$^rnn/gru_cell/gates/bias/Adam/Assign&^rnn/gru_cell/gates/bias/Adam_1/Assign^rnn/gru_cell/gates/bias/Assign&^rnn/gru_cell/gates/kernel/Adam/Assign(^rnn/gru_cell/gates/kernel/Adam_1/Assign!^rnn/gru_cell/gates/kernel/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
т
save/SaveV2/tensor_namesConst*Х
valueЛBИBbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Brnn/gru_cell/candidate/biasB rnn/gru_cell/candidate/bias/AdamB"rnn/gru_cell/candidate/bias/Adam_1Brnn/gru_cell/candidate/kernelB"rnn/gru_cell/candidate/kernel/AdamB$rnn/gru_cell/candidate/kernel/Adam_1Brnn/gru_cell/gates/biasBrnn/gru_cell/gates/bias/AdamBrnn/gru_cell/gates/bias/Adam_1Brnn/gru_cell/gates/kernelBrnn/gru_cell/gates/kernel/AdamB rnn/gru_cell/gates/kernel/Adam_1*
dtype0*
_output_shapes
:
Л
save/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
¤
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_power
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1rnn/gru_cell/candidate/bias rnn/gru_cell/candidate/bias/Adam"rnn/gru_cell/candidate/bias/Adam_1rnn/gru_cell/candidate/kernel"rnn/gru_cell/candidate/kernel/Adam$rnn/gru_cell/candidate/kernel/Adam_1rnn/gru_cell/gates/biasrnn/gru_cell/gates/bias/Adamrnn/gru_cell/gates/bias/Adam_1rnn/gru_cell/gates/kernelrnn/gru_cell/gates/kernel/Adam rnn/gru_cell/gates/kernel/Adam_1*"
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
х
save/RestoreV2/tensor_namesConst*Х
valueЛBИBbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Brnn/gru_cell/candidate/biasB rnn/gru_cell/candidate/bias/AdamB"rnn/gru_cell/candidate/bias/Adam_1Brnn/gru_cell/candidate/kernelB"rnn/gru_cell/candidate/kernel/AdamB$rnn/gru_cell/candidate/kernel/Adam_1Brnn/gru_cell/gates/biasBrnn/gru_cell/gates/bias/AdamBrnn/gru_cell/gates/bias/Adam_1Brnn/gru_cell/gates/kernelBrnn/gru_cell/gates/kernel/AdamB rnn/gru_cell/gates/kernel/Adam_1*
dtype0*
_output_shapes
:
О
save/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
я
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
Ы
save/AssignAssignbeta1_powersave/RestoreV2*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
Я
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
в
save/Assign_2Assign
dense/biassave/RestoreV2:2*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(
з
save/Assign_3Assigndense/bias/Adamsave/RestoreV2:3*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
й
save/Assign_4Assigndense/bias/Adam_1save/RestoreV2:4*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(
л
save/Assign_5Assigndense/kernelsave/RestoreV2:5*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	О*
use_locking(
░
save/Assign_6Assigndense/kernel/Adamsave/RestoreV2:6*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	О*
use_locking(
▓
save/Assign_7Assigndense/kernel/Adam_1save/RestoreV2:7*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	О
┼
save/Assign_8Assignrnn/gru_cell/candidate/biassave/RestoreV2:8*
use_locking(*
T0*.
_class$
" loc:@rnn/gru_cell/candidate/bias*
validate_shape(*
_output_shapes	
:О
╩
save/Assign_9Assign rnn/gru_cell/candidate/bias/Adamsave/RestoreV2:9*
use_locking(*
T0*.
_class$
" loc:@rnn/gru_cell/candidate/bias*
validate_shape(*
_output_shapes	
:О
╬
save/Assign_10Assign"rnn/gru_cell/candidate/bias/Adam_1save/RestoreV2:10*
validate_shape(*
_output_shapes	
:О*
use_locking(*
T0*.
_class$
" loc:@rnn/gru_cell/candidate/bias
╨
save/Assign_11Assignrnn/gru_cell/candidate/kernelsave/RestoreV2:11*
use_locking(*
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
validate_shape(* 
_output_shapes
:
ЭО
╒
save/Assign_12Assign"rnn/gru_cell/candidate/kernel/Adamsave/RestoreV2:12*
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
validate_shape(* 
_output_shapes
:
ЭО*
use_locking(
╫
save/Assign_13Assign$rnn/gru_cell/candidate/kernel/Adam_1save/RestoreV2:13*
use_locking(*
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
validate_shape(* 
_output_shapes
:
ЭО
┐
save/Assign_14Assignrnn/gru_cell/gates/biassave/RestoreV2:14*
validate_shape(*
_output_shapes	
:Ь*
use_locking(*
T0**
_class 
loc:@rnn/gru_cell/gates/bias
─
save/Assign_15Assignrnn/gru_cell/gates/bias/Adamsave/RestoreV2:15*
use_locking(*
T0**
_class 
loc:@rnn/gru_cell/gates/bias*
validate_shape(*
_output_shapes	
:Ь
╞
save/Assign_16Assignrnn/gru_cell/gates/bias/Adam_1save/RestoreV2:16*
T0**
_class 
loc:@rnn/gru_cell/gates/bias*
validate_shape(*
_output_shapes	
:Ь*
use_locking(
╚
save/Assign_17Assignrnn/gru_cell/gates/kernelsave/RestoreV2:17*
use_locking(*
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
validate_shape(* 
_output_shapes
:
ЭЬ
═
save/Assign_18Assignrnn/gru_cell/gates/kernel/Adamsave/RestoreV2:18*
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
validate_shape(* 
_output_shapes
:
ЭЬ*
use_locking(
╧
save/Assign_19Assign rnn/gru_cell/gates/kernel/Adam_1save/RestoreV2:19*
use_locking(*
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
validate_shape(* 
_output_shapes
:
ЭЬ
р
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Ж
save_1/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_080461fbeaeb4daf9064e77421bbe76f/part
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
Е
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
ф
save_1/SaveV2/tensor_namesConst*Х
valueЛBИBbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Brnn/gru_cell/candidate/biasB rnn/gru_cell/candidate/bias/AdamB"rnn/gru_cell/candidate/bias/Adam_1Brnn/gru_cell/candidate/kernelB"rnn/gru_cell/candidate/kernel/AdamB$rnn/gru_cell/candidate/kernel/Adam_1Brnn/gru_cell/gates/biasBrnn/gru_cell/gates/bias/AdamBrnn/gru_cell/gates/bias/Adam_1Brnn/gru_cell/gates/kernelBrnn/gru_cell/gates/kernel/AdamB rnn/gru_cell/gates/kernel/Adam_1*
dtype0*
_output_shapes
:
Н
save_1/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
П
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta2_power
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1rnn/gru_cell/candidate/bias rnn/gru_cell/candidate/bias/Adam"rnn/gru_cell/candidate/bias/Adam_1rnn/gru_cell/candidate/kernel"rnn/gru_cell/candidate/kernel/Adam$rnn/gru_cell/candidate/kernel/Adam_1rnn/gru_cell/gates/biasrnn/gru_cell/gates/bias/Adamrnn/gru_cell/gates/bias/Adam_1rnn/gru_cell/gates/kernelrnn/gru_cell/gates/kernel/Adam rnn/gru_cell/gates/kernel/Adam_1*"
dtypes
2
Щ
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
г
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
T0*

axis *
N*
_output_shapes
:
Г
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
В
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
ч
save_1/RestoreV2/tensor_namesConst*Х
valueЛBИBbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Brnn/gru_cell/candidate/biasB rnn/gru_cell/candidate/bias/AdamB"rnn/gru_cell/candidate/bias/Adam_1Brnn/gru_cell/candidate/kernelB"rnn/gru_cell/candidate/kernel/AdamB$rnn/gru_cell/candidate/kernel/Adam_1Brnn/gru_cell/gates/biasBrnn/gru_cell/gates/bias/AdamBrnn/gru_cell/gates/bias/Adam_1Brnn/gru_cell/gates/kernelBrnn/gru_cell/gates/kernel/AdamB rnn/gru_cell/gates/kernel/Adam_1*
dtype0*
_output_shapes
:
Р
!save_1/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ў
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*"
dtypes
2*d
_output_shapesR
P::::::::::::::::::::
Я
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@dense/bias
г
save_1/Assign_1Assignbeta2_powersave_1/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@dense/bias
ж
save_1/Assign_2Assign
dense/biassave_1/RestoreV2:2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense/bias
л
save_1/Assign_3Assigndense/bias/Adamsave_1/RestoreV2:3*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense/bias
н
save_1/Assign_4Assigndense/bias/Adam_1save_1/RestoreV2:4*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense/bias
п
save_1/Assign_5Assigndense/kernelsave_1/RestoreV2:5*
validate_shape(*
_output_shapes
:	О*
use_locking(*
T0*
_class
loc:@dense/kernel
┤
save_1/Assign_6Assigndense/kernel/Adamsave_1/RestoreV2:6*
validate_shape(*
_output_shapes
:	О*
use_locking(*
T0*
_class
loc:@dense/kernel
╢
save_1/Assign_7Assigndense/kernel/Adam_1save_1/RestoreV2:7*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	О
╔
save_1/Assign_8Assignrnn/gru_cell/candidate/biassave_1/RestoreV2:8*
T0*.
_class$
" loc:@rnn/gru_cell/candidate/bias*
validate_shape(*
_output_shapes	
:О*
use_locking(
╬
save_1/Assign_9Assign rnn/gru_cell/candidate/bias/Adamsave_1/RestoreV2:9*
use_locking(*
T0*.
_class$
" loc:@rnn/gru_cell/candidate/bias*
validate_shape(*
_output_shapes	
:О
╥
save_1/Assign_10Assign"rnn/gru_cell/candidate/bias/Adam_1save_1/RestoreV2:10*
use_locking(*
T0*.
_class$
" loc:@rnn/gru_cell/candidate/bias*
validate_shape(*
_output_shapes	
:О
╘
save_1/Assign_11Assignrnn/gru_cell/candidate/kernelsave_1/RestoreV2:11*
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel*
validate_shape(* 
_output_shapes
:
ЭО*
use_locking(
┘
save_1/Assign_12Assign"rnn/gru_cell/candidate/kernel/Adamsave_1/RestoreV2:12*
validate_shape(* 
_output_shapes
:
ЭО*
use_locking(*
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel
█
save_1/Assign_13Assign$rnn/gru_cell/candidate/kernel/Adam_1save_1/RestoreV2:13*
validate_shape(* 
_output_shapes
:
ЭО*
use_locking(*
T0*0
_class&
$"loc:@rnn/gru_cell/candidate/kernel
├
save_1/Assign_14Assignrnn/gru_cell/gates/biassave_1/RestoreV2:14*
validate_shape(*
_output_shapes	
:Ь*
use_locking(*
T0**
_class 
loc:@rnn/gru_cell/gates/bias
╚
save_1/Assign_15Assignrnn/gru_cell/gates/bias/Adamsave_1/RestoreV2:15*
validate_shape(*
_output_shapes	
:Ь*
use_locking(*
T0**
_class 
loc:@rnn/gru_cell/gates/bias
╩
save_1/Assign_16Assignrnn/gru_cell/gates/bias/Adam_1save_1/RestoreV2:16*
validate_shape(*
_output_shapes	
:Ь*
use_locking(*
T0**
_class 
loc:@rnn/gru_cell/gates/bias
╠
save_1/Assign_17Assignrnn/gru_cell/gates/kernelsave_1/RestoreV2:17*
use_locking(*
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
validate_shape(* 
_output_shapes
:
ЭЬ
╤
save_1/Assign_18Assignrnn/gru_cell/gates/kernel/Adamsave_1/RestoreV2:18*
use_locking(*
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
validate_shape(* 
_output_shapes
:
ЭЬ
╙
save_1/Assign_19Assign rnn/gru_cell/gates/kernel/Adam_1save_1/RestoreV2:19*
use_locking(*
T0*,
_class"
 loc:@rnn/gru_cell/gates/kernel*
validate_shape(* 
_output_shapes
:
ЭЬ
М
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard "B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"
train_op

Adam"гR
while_contextСRОR
ЛR
rnn/while/while_context *rnn/while/LoopCond:02rnn/while/Merge:0:rnn/while/Identity:0Brnn/while/Exit:0Brnn/while/Exit_1:0Brnn/while/Exit_2:0Brnn/while/Exit_3:0Bgradients/f_count_2:0J╟O
gradients/Add/y:0
gradients/Add:0
gradients/Merge:0
gradients/Merge:1
gradients/NextIteration:0
gradients/Switch:0
gradients/Switch:1
gradients/f_count:0
gradients/f_count_1:0
gradients/f_count_2:0
Vgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
\gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
Vgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
;gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/Enter:0
Agradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/StackPushV2:0
;gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/f_acc:0
9gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/Enter:0
?gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/StackPushV2:0
9gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/f_acc:0
=gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/Enter:0
Cgradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/StackPushV2:0
=gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/f_acc:0
Cgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/Enter:0
Egradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/Enter_1:0
Igradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPushV2:0
Kgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/StackPushV2_1:0
Cgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/f_acc:0
Egradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/f_acc_1:0
-gradients/rnn/while/gru_cell/add_grad/Shape:0
/gradients/rnn/while/gru_cell/add_grad/Shape_1:0
2gradients/rnn/while/gru_cell/concat_1_grad/Shape:0
9gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/Enter:0
;gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/Enter_1:0
?gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPushV2:0
Agradients/rnn/while/gru_cell/concat_1_grad/ShapeN/StackPushV2_1:0
9gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/f_acc:0
;gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/f_acc_1:0
0gradients/rnn/while/gru_cell/concat_grad/Shape:0
Egradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/Enter:0
Ggradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0
Kgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
Mgradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1:0
Egradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0
Ggradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0
3gradients/rnn/while/gru_cell/mul_1_grad/Mul/Enter:0
9gradients/rnn/while/gru_cell/mul_1_grad/Mul/StackPushV2:0
3gradients/rnn/while/gru_cell/mul_1_grad/Mul/f_acc:0
5gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/Enter:0
;gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/StackPushV2:0
5gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/f_acc:0
/gradients/rnn/while/gru_cell/mul_1_grad/Shape:0
1gradients/rnn/while/gru_cell/mul_1_grad/Shape_1:0
Egradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/Enter:0
Ggradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0
Kgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
Mgradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1:0
Egradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0
Ggradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0
3gradients/rnn/while/gru_cell/mul_2_grad/Mul/Enter:0
9gradients/rnn/while/gru_cell/mul_2_grad/Mul/StackPushV2:0
3gradients/rnn/while/gru_cell/mul_2_grad/Mul/f_acc:0
5gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/Enter:0
;gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/StackPushV2:0
5gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/f_acc:0
/gradients/rnn/while/gru_cell/mul_2_grad/Shape:0
1gradients/rnn/while/gru_cell/mul_2_grad/Shape_1:0
Cgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/Enter:0
Egradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/Enter_1:0
Igradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPushV2:0
Kgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
Cgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/f_acc:0
Egradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0
3gradients/rnn/while/gru_cell/mul_grad/Mul_1/Enter:0
9gradients/rnn/while/gru_cell/mul_grad/Mul_1/StackPushV2:0
3gradients/rnn/while/gru_cell/mul_grad/Mul_1/f_acc:0
-gradients/rnn/while/gru_cell/mul_grad/Shape:0
/gradients/rnn/while/gru_cell/mul_grad/Shape_1:0
Cgradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/Enter:0
Igradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/StackPushV2:0
Cgradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/f_acc:0
/gradients/rnn/while/gru_cell/sub_grad/Shape_1:0
rnn/Minimum:0
rnn/TensorArray:0
@rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
rnn/TensorArray_1:0
"rnn/gru_cell/candidate/bias/read:0
$rnn/gru_cell/candidate/kernel/read:0
rnn/gru_cell/gates/bias/read:0
 rnn/gru_cell/gates/kernel/read:0
rnn/strided_slice_1:0
rnn/while/Enter:0
rnn/while/Enter_1:0
rnn/while/Enter_2:0
rnn/while/Enter_3:0
rnn/while/Exit:0
rnn/while/Exit_1:0
rnn/while/Exit_2:0
rnn/while/Exit_3:0
rnn/while/Identity:0
rnn/while/Identity_1:0
rnn/while/Identity_2:0
rnn/while/Identity_3:0
rnn/while/Less/Enter:0
rnn/while/Less:0
rnn/while/Less_1/Enter:0
rnn/while/Less_1:0
rnn/while/LogicalAnd:0
rnn/while/LoopCond:0
rnn/while/Merge:0
rnn/while/Merge:1
rnn/while/Merge_1:0
rnn/while/Merge_1:1
rnn/while/Merge_2:0
rnn/while/Merge_2:1
rnn/while/Merge_3:0
rnn/while/Merge_3:1
rnn/while/NextIteration:0
rnn/while/NextIteration_1:0
rnn/while/NextIteration_2:0
rnn/while/NextIteration_3:0
rnn/while/Switch:0
rnn/while/Switch:1
rnn/while/Switch_1:0
rnn/while/Switch_1:1
rnn/while/Switch_2:0
rnn/while/Switch_2:1
rnn/while/Switch_3:0
rnn/while/Switch_3:1
#rnn/while/TensorArrayReadV3/Enter:0
%rnn/while/TensorArrayReadV3/Enter_1:0
rnn/while/TensorArrayReadV3:0
5rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
rnn/while/add/y:0
rnn/while/add:0
rnn/while/add_1/y:0
rnn/while/add_1:0
"rnn/while/gru_cell/BiasAdd/Enter:0
rnn/while/gru_cell/BiasAdd:0
$rnn/while/gru_cell/BiasAdd_1/Enter:0
rnn/while/gru_cell/BiasAdd_1:0
rnn/while/gru_cell/Const:0
!rnn/while/gru_cell/MatMul/Enter:0
rnn/while/gru_cell/MatMul:0
#rnn/while/gru_cell/MatMul_1/Enter:0
rnn/while/gru_cell/MatMul_1:0
rnn/while/gru_cell/Sigmoid:0
rnn/while/gru_cell/Tanh:0
rnn/while/gru_cell/add:0
 rnn/while/gru_cell/concat/axis:0
rnn/while/gru_cell/concat:0
"rnn/while/gru_cell/concat_1/axis:0
rnn/while/gru_cell/concat_1:0
rnn/while/gru_cell/mul:0
rnn/while/gru_cell/mul_1:0
rnn/while/gru_cell/mul_2:0
$rnn/while/gru_cell/split/split_dim:0
rnn/while/gru_cell/split:0
rnn/while/gru_cell/split:1
rnn/while/gru_cell/sub/x:0
rnn/while/gru_cell/sub:0n
5gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/f_acc:05gradients/rnn/while/gru_cell/mul_2_grad/Mul_1/Enter:0z
;gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/f_acc_1:0;gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/Enter_1:0n
5gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/f_acc:05gradients/rnn/while/gru_cell/mul_1_grad/Mul_1/Enter:0~
=gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/f_acc:0=gradients/rnn/while/gru_cell/Sigmoid_grad/SigmoidGrad/Enter:0J
"rnn/gru_cell/candidate/bias/read:0$rnn/while/gru_cell/BiasAdd_1/Enter:0К
Cgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/f_acc:0Cgradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/Enter:0К
Cgradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/f_acc:0Cgradients/rnn/while/gru_cell/sub_grad/BroadcastGradientArgs/Enter:0E
 rnn/gru_cell/gates/kernel/read:0!rnn/while/gru_cell/MatMul/Enter:0О
Egradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0Egradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/Enter:0О
Egradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0Egradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/Enter:0Т
Ggradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0Ggradients/rnn/while/gru_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0О
Egradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0Egradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/Enter_1:0J
rnn/TensorArray:05rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0j
3gradients/rnn/while/gru_cell/mul_grad/Mul_1/f_acc:03gradients/rnn/while/gru_cell/mul_grad/Mul_1/Enter:0Т
Ggradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0Ggradients/rnn/while/gru_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0j
3gradients/rnn/while/gru_cell/mul_1_grad/Mul/f_acc:03gradients/rnn/while/gru_cell/mul_1_grad/Mul/Enter:0K
$rnn/gru_cell/candidate/kernel/read:0#rnn/while/gru_cell/MatMul_1/Enter:0j
3gradients/rnn/while/gru_cell/mul_2_grad/Mul/f_acc:03gradients/rnn/while/gru_cell/mul_2_grad/Mul/Enter:0i
@rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0%rnn/while/TensorArrayReadV3/Enter_1:0v
9gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/f_acc:09gradients/rnn/while/gru_cell/MatMul_grad/MatMul_1/Enter:0D
rnn/gru_cell/gates/bias/read:0"rnn/while/gru_cell/BiasAdd/Enter:0v
9gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/f_acc:09gradients/rnn/while/gru_cell/concat_1_grad/ShapeN/Enter:0z
;gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/f_acc:0;gradients/rnn/while/gru_cell/MatMul_1_grad/MatMul_1/Enter:0К
Cgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/f_acc:0Cgradients/rnn/while/gru_cell/mul_grad/BroadcastGradientArgs/Enter:0)
rnn/Minimum:0rnn/while/Less_1/Enter:0:
rnn/TensorArray_1:0#rnn/while/TensorArrayReadV3/Enter:0░
Vgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0Vgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0/
rnn/strided_slice_1:0rnn/while/Less/Enter:0О
Egradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/f_acc_1:0Egradients/rnn/while/gru_cell/add_grad/BroadcastGradientArgs/Enter_1:0Rrnn/while/Enter:0Rrnn/while/Enter_1:0Rrnn/while/Enter_2:0Rrnn/while/Enter_3:0Rgradients/f_count_1:0Zrnn/strided_slice_1:0"и
	variablesЪЧ
Щ
rnn/gru_cell/gates/kernel:0 rnn/gru_cell/gates/kernel/Assign rnn/gru_cell/gates/kernel/read:026rnn/gru_cell/gates/kernel/Initializer/random_uniform:0
И
rnn/gru_cell/gates/bias:0rnn/gru_cell/gates/bias/Assignrnn/gru_cell/gates/bias/read:02+rnn/gru_cell/gates/bias/Initializer/Const:0
й
rnn/gru_cell/candidate/kernel:0$rnn/gru_cell/candidate/kernel/Assign$rnn/gru_cell/candidate/kernel/read:02:rnn/gru_cell/candidate/kernel/Initializer/random_uniform:0
Ш
rnn/gru_cell/candidate/bias:0"rnn/gru_cell/candidate/bias/Assign"rnn/gru_cell/candidate/bias/read:02/rnn/gru_cell/candidate/bias/Initializer/zeros:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
T
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:0
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
д
 rnn/gru_cell/gates/kernel/Adam:0%rnn/gru_cell/gates/kernel/Adam/Assign%rnn/gru_cell/gates/kernel/Adam/read:022rnn/gru_cell/gates/kernel/Adam/Initializer/zeros:0
м
"rnn/gru_cell/gates/kernel/Adam_1:0'rnn/gru_cell/gates/kernel/Adam_1/Assign'rnn/gru_cell/gates/kernel/Adam_1/read:024rnn/gru_cell/gates/kernel/Adam_1/Initializer/zeros:0
Ь
rnn/gru_cell/gates/bias/Adam:0#rnn/gru_cell/gates/bias/Adam/Assign#rnn/gru_cell/gates/bias/Adam/read:020rnn/gru_cell/gates/bias/Adam/Initializer/zeros:0
д
 rnn/gru_cell/gates/bias/Adam_1:0%rnn/gru_cell/gates/bias/Adam_1/Assign%rnn/gru_cell/gates/bias/Adam_1/read:022rnn/gru_cell/gates/bias/Adam_1/Initializer/zeros:0
┤
$rnn/gru_cell/candidate/kernel/Adam:0)rnn/gru_cell/candidate/kernel/Adam/Assign)rnn/gru_cell/candidate/kernel/Adam/read:026rnn/gru_cell/candidate/kernel/Adam/Initializer/zeros:0
╝
&rnn/gru_cell/candidate/kernel/Adam_1:0+rnn/gru_cell/candidate/kernel/Adam_1/Assign+rnn/gru_cell/candidate/kernel/Adam_1/read:028rnn/gru_cell/candidate/kernel/Adam_1/Initializer/zeros:0
м
"rnn/gru_cell/candidate/bias/Adam:0'rnn/gru_cell/candidate/bias/Adam/Assign'rnn/gru_cell/candidate/bias/Adam/read:024rnn/gru_cell/candidate/bias/Adam/Initializer/zeros:0
┤
$rnn/gru_cell/candidate/bias/Adam_1:0)rnn/gru_cell/candidate/bias/Adam_1/Assign)rnn/gru_cell/candidate/bias/Adam_1/read:026rnn/gru_cell/candidate/bias/Adam_1/Initializer/zeros:0
p
dense/kernel/Adam:0dense/kernel/Adam/Assigndense/kernel/Adam/read:02%dense/kernel/Adam/Initializer/zeros:0
x
dense/kernel/Adam_1:0dense/kernel/Adam_1/Assigndense/kernel/Adam_1/read:02'dense/kernel/Adam_1/Initializer/zeros:0
h
dense/bias/Adam:0dense/bias/Adam/Assigndense/bias/Adam/read:02#dense/bias/Adam/Initializer/zeros:0
p
dense/bias/Adam_1:0dense/bias/Adam_1/Assigndense/bias/Adam_1/read:02%dense/bias/Adam_1/Initializer/zeros:0"╞
trainable_variablesол
Щ
rnn/gru_cell/gates/kernel:0 rnn/gru_cell/gates/kernel/Assign rnn/gru_cell/gates/kernel/read:026rnn/gru_cell/gates/kernel/Initializer/random_uniform:0
И
rnn/gru_cell/gates/bias:0rnn/gru_cell/gates/bias/Assignrnn/gru_cell/gates/bias/read:02+rnn/gru_cell/gates/bias/Initializer/Const:0
й
rnn/gru_cell/candidate/kernel:0$rnn/gru_cell/candidate/kernel/Assign$rnn/gru_cell/candidate/kernel/read:02:rnn/gru_cell/candidate/kernel/Initializer/random_uniform:0
Ш
rnn/gru_cell/candidate/bias:0"rnn/gru_cell/candidate/bias/Assign"rnn/gru_cell/candidate/bias/read:02/rnn/gru_cell/candidate/bias/Initializer/zeros:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
T
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:0"#
	summaries

loss:0

accuracy:0*╨
serving_default╝
;
inputs1
Placeholder:0                  
0
target&
Placeholder_1:0         /
predictions 
	Softmax:0         tensorflow/serving/predict