#!/usr/bin/env python

def unpackerBody(bit, packSize, kGroupSize):
    lines = []
    if bit == 0:
        lines.append("\n\tif (!IsRiceCoding) { // For NewPFor and OptPFor.")
        lines.append("\n\t\tmemset(out, 0, sizeof(uint32_t) * %d);" % packSize)
        lines.append("\n\t}")
        lines.append("\n\telse { // For Rice and OptRice.")
        lines.append("\n\t\tmemcpy(out, quotient, sizeof(uint32_t) * %d);" % packSize);
        lines.append("\n\t}")
        return "".join(lines)
    if bit == 32:
        return "\n\tmemcpy(out, in, sizeof(uint32_t) * %d);" % packSize

    for i in range(0, packSize / kGroupSize):
        idx = (i * bit) >> 5      # word index
        shift = (i * bit) & 0x1f  # number of bits used in current word
        mask = "& 0x%0*x " % (2 * ((bit-1)/8) + 2, (1<<bit)-1)  # hexadecimal mask with padding zero 

        for j in range(0, kGroupSize):
            lines.append("\n\tout[%d] = ( in[%d] >> %d ) %s;"  % (kGroupSize*i+j, kGroupSize*idx+j, shift, mask if bit < 32 - shift else ""))
        lines.append("\n" if kGroupSize > 1 or bit == 32 - shift else "")

        if (bit > 32 - shift):  # codeword spanning across consecutive groups 
            lines.append("\n" if kGroupSize == 1 else "")
            for j in range(0, kGroupSize):
                lines.append("\n\tout[%d] |= ( in[%d] << ( 32 - %d ) ) %s;" % (kGroupSize*i+j, kGroupSize*(idx+1)+j, shift, mask))
            lines.append("\n" if kGroupSize > 1 else "")
    lines.append("\n\tif (IsRiceCoding) { // For Rice and OptRice.")
    lines.append("\n\t\tfor (uint32_t i = 0; i < %d; ++i)" % packSize)
    lines.append("\n\t\t\tout[i] |= quotient[i] << %d;" % bit)
    lines.append("\n\t}")

    return "".join(lines)


def packerBody(bit, packSize, kGroupSize, withoutmask):
	if (bit == 32):
		return "\n\tmemcpy(out, in, %d * sizeof(uint32_t));\n" % packSize

	lines = []
	for i in range(0, packSize / kGroupSize):
		idx = (i * bit) >> 5      # word index
		shift = (i * bit) & 0x1f  # number of bits used in current word
		mask = "& 0x%0*x " % (2 * ((bit-1)/8) + 2, (1<<bit)-1)

		for j in range(0, kGroupSize):
			if withoutmask or bit >= 32 - shift:
				lines.append("\n\tout[%d] %s in[%d] << %d ;" % (kGroupSize*idx+j, "=" if shift == 0 else "|=", kGroupSize*i+j, shift))
			else:
				lines.append("\n\tout[%d] %s ( in[%d] %s) << %d ;" % (kGroupSize*idx+j, "=" if shift == 0 else "|=", kGroupSize*i+j, mask, shift))
		lines.append("\n" if kGroupSize > 1 or bit == 32 - shift else "")

		if (bit > 32 - shift):  # codeword spanning across consecutive groups
			lines.append("\n" if kGroupSize == 1 else "")
			for j in range(0, kGroupSize):
				if withoutmask:
					lines.append("\n\tout[%d] = in[%d] >> ( 32 - %d ) ;" % (kGroupSize*(idx+1)+j, kGroupSize*i+j, shift))
				else:
					lines.append("\n\tout[%d] = ( in[%d] %s) >> ( 32 - %d ) ;" % (kGroupSize*(idx+1)+j, kGroupSize*i+j, mask, shift))
			lines.append("\n" if kGroupSize > 1 else "")

	return "".join(lines)


# packSize: number of values to be unpacked for each call
# kGroupSize: 1 for horizontal unpackers
#             4/8 for vertical unpackers
def generate(codecName, className, packSize, kGroupSize):
    fileName = "../include/%s.h" % codecName
    target = open(fileName, 'w')
    format = "horizontal" if kGroupSize == 1 else "vertical"

    lines = []
    lines.append("#ifndef CODECS_%s_H_" % codecName.upper())
    lines.append("\n#define CODECS_%s_H_\n" % codecName.upper())
    for bit in range(0, 33):
        lines.append("\n// %d-bit" % bit) # comment
        lines.append("\ntemplate <bool IsRiceCoding>")
        lines.append("\nvoid %s<Scalar, IsRiceCoding>::%sunpack_c%d(uint32_t *  __restrict__  out,\n\t\tconst uint32_t *  __restrict__  in) {" % (className, format, bit)) # prototype
        lines.append(unpackerBody(bit, packSize, kGroupSize))  # function body
        lines.append("\n}\n")

    for bit in range(1, 33):
        lines.append("\n// %d-bit" % bit)
        if format == "vertical":
            lines.append("\ntemplate <bool IsRiceCoding>")
            lines.append("\nvoid %s<Scalar, IsRiceCoding>::%spackwithoutmask_c%d(uint32_t *  __restrict__  out,\n\t\tconst uint32_t * __restrict__  in) {" % (className, format, bit))
        else:
            lines.append("\ntemplate <typename InstructionSet, bool IsRiceCoding>")
            lines.append("\nvoid %sBase<InstructionSet, IsRiceCoding>::%spackwithoutmask_c%d(uint32_t *  __restrict__  out,\n\t\tconst uint32_t * __restrict__  in) {" % (className, format, bit))
        lines.append(packerBody(bit, packSize, kGroupSize, True))
        lines.append("}\n")

    for bit in range(1, 33):
        lines.append("\n// %d-bit" % bit)
        if format == "vertical":
            lines.append("\ntemplate <bool IsRiceCoding>")
            lines.append("\nvoid %s<Scalar, IsRiceCoding>::%spack_c%d(uint32_t *  __restrict__  out,\n\t\tconst uint32_t * __restrict__  in) {" % (className, format, bit))
        else:
            lines.append("\ntemplate <typename InstructionSet, bool IsRiceCoding>")
            lines.append("\nvoid %sBase<InstructionSet, IsRiceCoding>::%spack_c%d(uint32_t *  __restrict__  out,\n\t\tconst uint32_t * __restrict__  in) {" % (className, format, bit))
        lines.append(packerBody(bit, packSize, kGroupSize, False))
        lines.append("}\n")

    lines.append("\n#endif /* CODECS_%s_H_ */\n" % codecName.upper())
    target.write("".join(lines))

    target.close()

def main():
	generate("HorScalarUnpacker", "HorUnpacker", 64, 1) # HorizontalScalarUnpacker.h

if __name__ == "__main__":
	main()

