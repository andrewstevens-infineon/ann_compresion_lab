import sys
import struct
from enum import Enum
import numpy as np
import math
import flatbuffers
from typing import Optional


import ifx_tflite.Model
import ifx_tflite.QuantizationDetails
import ifx_tflite.CustomQuantization
from ifx_tflite.Tensor import TensorT
from ifx_tflite.BuiltinOperator import BuiltinOperator as OpType
from ifx_tflite.TensorType import TensorType as TType
from ifx_tflite.QuantizationDetails import QuantizationDetails as QDetails


PACKED_SUB8BIT_UNIFORM_DETAILS_MAGIC = 0xa4592d92
SPARSITY_FORMATS = ["cpu", "rival2"]


_SPARSE_RUN_DIMS  = {
    OpType.FULLY_CONNECTED : 1,
    # Currently only support unpacking entire filter...
    OpType.CONV_2D : 3, 
    # Caution the internal tflite dw_conv2D filter tensor dimensions
    # are NOT the tensorflow ops (NHWC) layout of
    # [filter_height, filter_width, in_channels, channel_multiplier]
    # for some reason tflite has 
    # [1, filter_height, filter_widths, in_channels * channel_multiplier]
    # 
    # Currently only support unpacking entire filter...
    OpType.DEPTHWISE_CONV_2D: 3
}

_SAME_ZP_DIMS  = {
    OpType.FULLY_CONNECTED : 1,
    # Currently only support unpacking entire filter...
    OpType.CONV_2D : 3, 

    # Caution the internal tflite dw_conv2D filter tensor dimensions
    # are NOT the tensorflow ops (NHWC) layout of
    # [filter_height, filter_width, in_channels, channel_multiplier]
    # for some reason tflite has 
    # [1, filter_height, filter_widths, in_channels * channel_multiplier]
    # Different Zero-point for each value in minor dimension!
    OpType.DEPTHWISE_CONV_2D: 0
}



_PACKING_RUN_DIMS  = {
    OpType.FULLY_CONNECTED : 1,
    # Currently only support unpacking entire filter...
    OpType.CONV_2D : 4, 
    # Caution the internal tflite dw_conv2D filter tensor dimensions
    # are NOT the tensorflow ops (NHWC) layout of
    # [filter_height, filter_width, in_channels, channel_multiplier]
    # for some reason tflite has 
    # [1, filter_height, filter_widths, in_channels * channel_multiplier]
    # 
    # Currently only support unpacking channel-wise (enables a reasonably
    # efficient naive implementation of RAM optmized version that does require
    # a buffer for unpacked weights).
    OpType.DEPTHWISE_CONV_2D: 1
}

#
# Ouch... flatbuffer python binding generator does not seem to generate
# an equivalent of ModelIdentifier present in the C++ bindings.
#
_TFLITE_FILE_IDENTIFIER = b"TFL3"

# 
def loadModel(model_flatbuf):

    """  Load tflite flatbuffer file / bytes to object tree.
    """

    if type(model_flatbuf) is bytes:
        buf = bytearray(model_flatbuf)
    else:
        with open(model_flatbuf, "rb") as f:
            buf = bytearray(f.read())

    model = ifx_tflite.Model.Model.GetRootAsModel(buf, 0)
    return ifx_tflite.Model.ModelT.InitFromObj(model)


def modelFlatbuffer(modelT):
    """Generate flatbuffer bytearray representation of tflite model object tree 
    """
    b = flatbuffers.Builder(1024)
    model_offset = modelT.Pack(b)
    b.Finish(model_offset, file_identifier=_TFLITE_FILE_IDENTIFIER)

    return b.Output()


def saveModel(filename, modelT):
    """Save tflite model object tree as tflite flatbuffer file.
    """

    model_fb = modelFlatbuffer(modelT)
    with open(filename, "wb") as f:
        f.write(model_fb)


class SparseTypes:

    DENSE = 0
    WORD_RUN_LENGTHS = 1

def setQuantizationDetails(t : TensorT, bits_per_item, container_bits, packed_minor_dims, sparse_type):

    # Serialize custom quantization data.
    data = struct.pack("I", PACKED_SUB8BIT_UNIFORM_DETAILS_MAGIC)
    data += struct.pack("B", bits_per_item)
    data += struct.pack("B", container_bits)
    data += struct.pack("B", packed_minor_dims)
    data += struct.pack("B", sparse_type)
    data += b"\x00\x00\x00\x00"

    customQuant = ifx_tflite.CustomQuantization.CustomQuantizationT()
    customQuant.custom = [c for c in data]
    t.quantization.detailsType = QDetails.CustomQuantization
    t.quantization.details = customQuant


def selectPackingFormat(bits_required_to_code : int, op_code : int):

    """
    Choose optimal tflite(u)-RMF custom quantization format parameters
    
    Currently choice is trivial as no alternatives available.

    """
    bits_per_item = bits_required_to_code
    packed_minor_dims = _PACKING_RUN_DIMS[op_code]

    if bits_required_to_code == 4:
        container_bits = 8
    elif bits_required_to_code == 5:
        container_bits = 16
    elif bits_required_to_code == 6:
        container_bits = 32
    else:
        return None

    return (bits_per_item, container_bits, packed_minor_dims)


def tryPackTensor(cnst_tensor, tfl_model, bits_required_to_code : int, op_code : int, verbosity : int):

    """
    Attempt narrow bit-width packing of constant tensor to save ROM space.

    :param cnst_tensor:   Constant INT8 quantized weight tensor to pack
    :param tfl_model:  tfl_model containing tensor to pack
    :param bits_required_to_code:  Minimal bits required to code values present in tensor
    :param op_code:  Operator take tensor as weights input.
    """

    if  verbosity >= 2:
        print( "TENSOR PACKING CANDIDATE", cnst_tensor.name, bits_required_to_code, "BITS")
    format_info = selectPackingFormat(bits_required_to_code, op_code)

    if format_info is None:
        print("WARNING: ", cnst_tensor.name, " values suggest unsupported bit-width", bits_required_to_code)
        return

    bits_per_item, container_bits, packed_minor_dims = format_info

    # Determine run length.
    packing_run_length = np.prod(cnst_tensor.shape[-packed_minor_dims:])

    # Pack tensor data.
    packed_data = bytes()
    buf = 0
    bits_in_container = 0
    mask = (1 << bits_per_item) - 1

    fmtLookup = {8: "B", 16: "H", 32: "I"}
    packing_fmt = fmtLookup[container_bits]
    for i, d in enumerate(tfl_model.buffers[cnst_tensor.buffer].data):
        buf |= (d & mask) << bits_in_container
        bits_in_container += bits_per_item
        # Flush full or last container.
        if (bits_in_container + bits_per_item > container_bits or
                i % packing_run_length == packing_run_length - 1):
            packed_data += struct.pack(packing_fmt, buf)

            bits_in_container = 0
            buf = 0

    assert bits_in_container == 0, "leftover data"
    if len(packed_data) > len(tfl_model.buffers[cnst_tensor.buffer].data):
        if  verbosity >= 2:
            print("Warning: not packing tensor:", cnst_tensor.name, " as this would increased tensor size")
        return

    setQuantizationDetails(cnst_tensor, bits_per_item, container_bits, packed_minor_dims, SparseTypes.DENSE)
    tfl_model.buffers[cnst_tensor.buffer].data = [c for c in packed_data]


def round_to_multiple(x, multiple):

    return (x+multiple-1)//multiple * multiple

def sameZeroPointRunLen(op_code, shape):

    same_zp_dims = _SAME_ZP_DIMS.get(op_code, None)
    if same_zp_dims is None:
        return None
    else:
        return np.prod(shape[-same_zp_dims:], dtype=np.int)


def sparseRunLen(op_code, shape):

    sparse_run_dims = _SPARSE_RUN_DIMS.get(op_code, None)
    if sparse_run_dims is None:
        return None
    else:
        return np.prod(shape[-sparse_run_dims:])


def trySparseTensor(cnst_tensor : TensorT, tfl_model, op_code, sparsity_format :str, verbosity : int):

    """
    Attempt sparse coding tensor to save ROM space.

    :param cnst_tensor:   Constant INT8 quantized weight tensor to pack
    :param tfl_model:  tfl_model containing tensor to pack
    :param op_code:  Operator take tensor as weights input.
    :param sparsity_format:  Format to use for sparse coding.
    """

    if  verbosity >= 2:
        print( "SPARSE TENSOR CANDIDATE", cnst_tensor.name)
    quantization = cnst_tensor.quantization

    # Check we have supported quantization granularity - currently
    # only per-layer quantization.
    if quantization.zeroPoint is  None:
        return
    zero_points = quantization.zeroPoint
    zero_points_len = len(quantization.zeroPoint)



    dense_tensor_data = tfl_model.buffers[cnst_tensor.buffer].data
    if dense_tensor_data is None:
        return


    zp_run_len = sameZeroPointRunLen(op_code, cnst_tensor.shape)
    if zp_run_len is None:
        return

    # This is actually only needed for rival2 format...
    sparse_run_len = sparseRunLen(op_code, cnst_tensor.shape)
    assert sparse_run_len is not None

    
    dense_data_len = len(dense_tensor_data)
    assert dense_data_len % zp_run_len == 0, "Dense data length must be multiple of zero-point run length"
    
    if zero_points_len == 1:
        zeros = zero_points[0]  # Rely on broadcasting...
    else:
        # For Depthwise conv ... different zero-points of different filters
        # vary across minor (input_channels * channel_multipler) dimension of filter tensor 
        zeros = [zero_points[i//zp_run_len%zero_points_len] for i in range(0, dense_data_len)]

    zero_map = dense_tensor_data == zeros
    num_zeros = np.count_nonzero(zero_map)

    # Check it is really worth using a sparse representation
    if num_zeros < len(dense_tensor_data)/16+1:
        print("    TENSOR INSUFFICIENTLY SPARSE - skipping")
        return

    # Create packed-word sparsity map
    # Bits are packed into 32-bit words to enable a word-aligned memory layout in the flat-buffer.
    bitsPerItem = 8
    containerBits = 8
    packedMinorDims = 0


    # Sparsity map round to mutliple full words to ensure alginable to word boundaries.
    sparsity_map_align_len = max(1, ((dense_data_len+7) // 8))  # Length  bytes
    sparsity_bitmap = bytes()
    nonzero_weights = bytes()

    # RiVAL2 input co-processor needs the number of non-zero elements in each
    # "run" (kernel footprint) of sparse data.

    if sparsity_format == "rival2":
        sparse_data_run_lens = bytes()


    bits_accum = 0

    if sparsity_format == "rival2":
        run_nz_elts = 0

    # Fill sparse data vector.   The word-aligned sparsity map (whose size is given by
    # tensor shape) comes first followed by the non-zero data elements
    # Flat-buffers are little-endian so we can simply write out
    # sparsity map bits to data bytes in-order.

    for elt_i, dense_elt in enumerate(dense_tensor_data):
        if not zero_map[elt_i]:
            bits_accum |= 1 << (elt_i%8)
            nonzero_weights += struct.pack("B",dense_elt)
            if sparsity_format == "rival2":
                run_nz_elts += 1
        if elt_i % 8 == 7:
            sparsity_bitmap += struct.pack("B",bits_accum)
            bits_accum = 0
        if sparsity_format == "rival2":
            # End of a sparse data run?
            if elt_i % sparse_run_len == sparse_run_len-1:
                sparse_data_run_lens += struct.pack("H", run_nz_elts)
                run_nz_elts = 0

    # Flush any remaining sparsity bits not yet written out
    if elt_i % 8 != 7:
        sparsity_bitmap += struct.pack("B", bits_accum)
    assert len(sparsity_bitmap) == sparsity_map_align_len, "sane bitmap len"
    assert elt_i % sparse_run_len == sparse_run_len - 1, "sane sparse_run_len"


    # Replace original buffer contents (dense tensor data) with the new
    # sparse data
    if sparsity_format == "rival2":
        buf_data = np.frombuffer(sparse_data_run_lens + sparsity_bitmap + nonzero_weights, dtype=np.uint8)
    else:
        buf_data = np.frombuffer(sparsity_bitmap + nonzero_weights, dtype=np.uint8)
    if (len(buf_data) > dense_data_len):
        if  verbosity >= 2:
            print("    TENSOR LARGER WHEN SPARSE - skipping")
    else:
        if  verbosity >= 1:
            print("    SPARSE TENSOR PACKED", cnst_tensor.name, num_zeros, "/", len(dense_tensor_data))
        setQuantizationDetails(cnst_tensor, bitsPerItem, containerBits, packedMinorDims, SparseTypes.WORD_RUN_LENGTHS)
        tfl_model.buffers[cnst_tensor.buffer].data = buf_data


def requiredQuantizedBits( tensor_data ):

    range = tensor_data.max() - tensor_data.min()
    bits = int(math.log2(max(range, 1)))+1
    return bits


# Packs all eligible tensors in the given model.
def packModel(tfl_model, sparsity_format, verbosity=0):

    if sparsity_format not in SPARSITY_FORMATS:
        raise ValueError(f"sparsity_format {sparsity_format} unsupported - not one of {SPARSITY_FORMATS}")
    tensorsToOps = {}

    for g in tfl_model.subgraphs:
        for op in g.operators:
            for tIndexList in [op.inputs, op.outputs, op.intermediates]:
                if tIndexList is None:
                    continue
                for i in tIndexList:
                    tensorsToOps.setdefault(i, []).append(op)

        # Get candidates for packing.
        for tIndex, t in enumerate(g.tensors):
            # Check we have a quantized tensor of supported type and
            # quantizatino
            if t.type not in [TType.UINT8, TType.INT8]:
                continue
            if not t.quantization:
                continue
            if t.quantization.details:
                continue

            # Check if all usages of this tensor support packing.
            packable = True
            for op in tensorsToOps[tIndex]:
                op_code = tfl_model.operatorCodes[op.opcodeIndex].builtinCode
                if op_code not in [OpType.FULLY_CONNECTED, OpType.CONV_2D, OpType.DEPTHWISE_CONV_2D]:
                    packable = False
            if not packable:
                continue

            # Check we have constant tensor
     
            if t.buffer == 0:
                continue
            tensor_data = tfl_model.buffers[t.buffer].data
            if tensor_data is None:
                continue

            # Flatbuffer buffer data is just
            if t.type in [TType.INT8]:
                tensor_data = tensor_data.astype(np.int8).astype(np.int32)
            # No point squashing tiny tensors...

            if tensor_data.size < 32:
                continue

            # Check we have supported bit-width
            req_bits = requiredQuantizedBits(tensor_data)
            if req_bits > 8:
                continue

            if req_bits < 8:
                # Sparse kernels only for 8-bit weights right now
                tryPackTensor(t, tfl_model, req_bits, op_code, verbosity)
            else:  # req_bits == 8
                trySparseTensor(t, tfl_model, op_code, sparsity_format, verbosity)


# Print whole model for debugging.
def printModel(m):
    import jsonpickle
    import yaml
    s = jsonpickle.encode(m)
    print(yaml.dump(yaml.load(s), indent=2))


def packTflFB(input_tflite_pathname : str, packed_tflite_pathname : str, 
              sparsity_format : Optional[str] = None,
              verbosity=1):

    """
    Provides a Python interface to the tfl-RMF packing utilty.

    The .tflite model in `input_tfl_fb` is read in.  Constant weight inputs
    to operators supporting tfl-rmf packed weight formats are identified.
    Those containing suitable narrow bit-width (4,5 or 6 bit) or sparse values 
    are repacked to use  tfl-rmf compressed tensor formats wherever whenever this
    would save memory.

    The resulting, smaller, .tflite model is wrtten out to `packed_tfl_fb`.

    :param input_tflite_pathname:   Input .tflite file 
    :param packed_tflite_pathname:  Output .tflite file using tfl-RMF compressed tensor formats.
    :param sparsity_format:  Sparse tensor packing format to be used. Defaults to software-optimized
    :param verbosity:  Verbosity of diagnostic output to stdout (0 = none, 1, packed tensors, 2 = full info)
    """

    if sparsity_format is None:
        sparsity_format = SPARSITY_FORMATS[0]
    
    if verbosity >= 1:
        print("PACKING:", input_tflite_pathname, "sparsity format:", sparsity_format)
    tfl_model = loadModel(input_tflite_pathname)
    packModel(tfl_model, sparsity_format, verbosity)
    saveModel(packed_tflite_pathname, tfl_model)


def pack(input_tfl_fb : bytes, sparsity_format : Optional[str] = None, verbosity=0):

    """
    Provides a Python interface to the tfl-RMF packing utilty.

    The .tflite model in `input_tfl_fb` is read in.  Constant weight inputs
    to operators supporting tfl-rmf packed weight formats are identified.
    Those containing suitable narrow bit-width (4,5 or 6 bit) or sparse values 
    are repacked to use  tfl-rmf compressed tensor formats wherever whenever this
    would save memory.

    The resulting, smaller, .tflite model is wrtten out to `packed_tfl_fb`.

    :param input_tfl_fb:   Input .tflite flatbuffer file 
    :param packed_tfl_fb:  Output .tflite file using tfl-RMF compressed tensor formats.
    :param sparsity_format:  Sparse tensor packing format to be used. Defaults to software-optimized
    :param verbosity:  Verbosity of diagnostic output to stdout (0 = none, 1, packed tensors, 2 = full info)
    """

    if sparsity_format is None:
        sparsity_format = SPARSITY_FORMATS[0]    
    tfl_model = loadModel(input_tfl_fb)
    packModel(tfl_model, sparsity_format, verbosity)
    return modelFlatbuffer(tfl_model)

