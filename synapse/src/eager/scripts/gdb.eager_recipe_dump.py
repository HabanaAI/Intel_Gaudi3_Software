#'''gdb command `dumprecipe <recipe_t*>` to print the contents of recipe_t.
#Intended to be used for comparison based debug.'''

from collections import defaultdict
from functools import partial
import gdb
import hashlib
import abc

class Utils:
    @staticmethod
    def ad(cmd, start_bit: int , count: int):
        assert 64 > start_bit >= 0 and count > 0 and 64 >= start_bit + count >= 0
        return (cmd >> start_bit) & ((1 << count) - 1)

    @staticmethod
    def ah(cmd, start_bit: int, count: int):
        return hex(Utils.ad(cmd, start_bit, count))

    @staticmethod
    def cmd_field_to_str(name, start_bit, count, cmd):
        max_width = len(hex((1 << count) - 1))
        return f'{name}: {Utils.ah(cmd,start_bit,count):<{max_width}}'

    @staticmethod
    def as_unsigned_char_ptr(char_ptr):
        tmp = gdb.lookup_type('unsigned char').pointer()
        return char_ptr.cast(tmp)

    @staticmethod
    def as_uint32_ptr(char_ptr):
        tmp = gdb.lookup_type('uint32_t').pointer()
        return char_ptr.cast(tmp)

    @staticmethod
    def as_void_ptr(char_ptr):
        void_ptr_type = gdb.lookup_type('void').pointer()
        return char_ptr.cast(void_ptr_type)

    @staticmethod
    def as_uint64_t_ptr(ptr):
        uintptr_t = gdb.lookup_type('uintptr_t')
        assert ptr.cast(uintptr_t) % 8 == 0, f"ptr 0x{ptr:x} isn't aligned to 8 bytes, casting to uint64_t might be a problem"
        uint64_t_ptr = gdb.lookup_type('uint64_t').pointer()
        return ptr.cast(uint64_t_ptr)

    @staticmethod
    def as_uint32_t_ptr(ptr):
        uintptr_t = gdb.lookup_type('uintptr_t')
        assert ptr.cast(uintptr_t) % 4 == 0, f"ptr 0x{ptr:x} isn't aligned to 4 bytes, casting to uint32_t might be a problem"
        uint32_t_ptr = gdb.lookup_type('uint32_t').pointer()
        return ptr.cast(uint32_t_ptr)

class HalReader(metaclass=abc.ABCMeta):
    def __init__(self, block_context):
        super().__init__()
        self.block = gdb.lookup_global_symbol(block_context).symtab.global_block()
        # those are the same for Gaudi3\Gaudi2 so we define it once here
        self.qman_cmd_opcode_to_str = {
            0x1 : "WREG_32",
            0x2 : "WREG_BULK",
            0x8 : "FENCE",
            0xa : "NOP",
            0x12 : "WREG_64_SHORT",
            0x13 : "WREG_64_LONG",
        }
        # those are the same for Gaudi3\Gaudi2 so we define it once here
        self.ecb_cmd_opcode_to_str = {
            0x0 : "ECB_CMD_LIST_SIZE",
            0x1 : "ECB_CMD_NOP",
            0x2 : "ECB_CMD_WD_FENCE_AND_EXE",
            0x3 : "ECB_CMD_SCHED_DMA",
            0x4 : "ECB_CMD_STATIC_DESC_V2",
        }

    def get_wd_type(self, type_name):
        return gdb.lookup_type(type_name, self.block)

    def ecb_cmd_to_op_code(self, cmd):
        op_code = Utils.ad(cmd,0,4)
        assert op_code in self.ecb_cmd_opcode_to_str, f"unsupported command op code {op_code}"
        return op_code

    def ecb_cmd_to_op_str(self, cmd):
        return self.ecb_cmd_opcode_to_str[self.ecb_cmd_to_op_code(cmd)]

    def qman_cmd_to_op_code(self, cmd):
        op_code = Utils.ad(cmd,56,5)
        assert op_code in self.qman_cmd_opcode_to_str, f"unsupported command op code {op_code}"
        return op_code

    def qman_cmd_to_op_str(self, cmd):
        return self.qman_cmd_opcode_to_str[self.qman_cmd_to_op_code(cmd)]

    @abc.abstractmethod
    def fmt_qman_cmd_fields(self, cmd):
        return None

    @abc.abstractmethod
    def fmt_ecb_cmd_fields(self, cmd):
        return None

    # following are the same for Gaudi2 and Gaudi3 so currently no need for divergence
    def get_wreg_32_reg(self, cmd):
        return Utils.ad(cmd,37,1)

    def get_wreg_32_offset_and_value(self, cmd):
        reg_offset = Utils.ad(cmd,40,16)
        value = Utils.ad(cmd,0,32)
        return reg_offset, value

    def get_wreg_bulk_size(self, cmd):
        return Utils.ad(cmd,0,16)

    def get_wreg_bulk_offset(self, cmd):
        return Utils.ad(cmd,40,16)

    def get_ecb_nop_padding(self, cmd):
        return Utils.ad(cmd,9,23)

    def get_ecb_sched_dma_addr_index(self, cmd):
        return Utils.ad(cmd, 5, 3)

    def get_ecb_sched_dma_size(self, cmd):
        return Utils.ad(cmd, 8, 10)

    def get_ecb_static_desc_v2_addr_index(self, cmd):
        return Utils.ad(cmd, 29, 3)

    def get_ecb_static_desc_v2_size(self, cmd):
        return Utils.ad(cmd, 16, 13)

class Gaudi2HalReader(HalReader):
    def __init__(self):
        super().__init__('Gaudi2Graph')

    def fmt_qman_cmd_fields(self, cmd):
        s = partial(Utils.cmd_field_to_str, cmd = cmd)
        op = self.qman_cmd_to_op_str(cmd)
        if op == 'WREG_32':
            return f'{s("MB",63,1)}, {s("SWTC",62,1)}, {s("EB",61,1)}, {s("OP",56,5)} ({"WREG 32":<15}), {s("REG_OFFSET",40,16)}, {s("RSV",38,2)}, {s("PRED",32,5)}, {s("REG_ID" if Utils.ad(cmd,37,1) == 1 else "VALUE",0,32)}'
        elif op == 'WREG_BULK':
            return f'{s("MB",63,1)}, {s("SWTC",62,1)}, {s("EB",61,1)}, {s("OP",56,5)} ({"WREG BULK":<15}), {s("REG_OFFSET",40,16)}, {s("PRED",32,5)}, {s("SIZE64",0,16)}'
        elif op == 'FENCE':
            return f'{s("MB",63,1)}, {s("SWTC",62,1)}, {s("EB",61,1)}, {s("OP",56,5)} ({"FENCE":<15}), {s("PRED",32,5)}, {s("ID",30,2)}, {s("TARGET_VAL",16,8)}, {s("DEC_VAL",0,4)}'
        elif op == 'NOP':
            return f'{s("MB",63,1)}, {s("SWTC",62,1)}, {s("EB",61,1)}, {s("OP",56,5)} ({"NOP":<15})'
        elif op == 'WREG_64_SHORT':
            return f'{s("MB",63,1)}, {s("SWTC",62,1)}, {s("EB",61,1)}, {s("OP",56,5)} ({"WREG_64_SHORT":<15}), {s("DREG_OFFSET",42,14)}, {s("BASE",37,4)}, {s("PRED",32,5)}, {s("REG_ID" if Utils.ad(cmd,41,1) == 1 else "OFFSET",0,32)}'
        elif op == 'WREG_64_LONG':
            return f'{s("MB",63,1)}, {s("SWTC",62,1)}, {s("EB",61,1)}, {s("OP",56,5)} ({"WREG_64_LONG":<15}), {s("DREG_OFFSET",42,14)}, {s("REL",41,1)}, {s("BASE",37,4)}, {s("PRED",32,5)}, {s("DW_ENABLE",0,2)}'

    def fmt_ecb_cmd_fields(self, cmd):
        s = partial(Utils.cmd_field_to_str, cmd = cmd)
        op = self.ecb_cmd_to_op_str(cmd)
        if op == 'ECB_CMD_LIST_SIZE':
            return f'{s("cmd_type", 0, 4)} ({"ECB_CMD_LIST_SIZE":24}), {s("yield", 4,1)}, {s("topology_start", 5, 1)}, {s("reserved", 6, 2)}, {s("list_size", 8, 24)}'
        elif op == 'ECB_CMD_NOP':
            return f'{s("cmd_type", 0, 4)} ({"ECB_CMD_NOP":24}), {s("yield", 4,1)}, {s("dma_completion", 5, 3)}, {s("switch_cq", 8, 1)}, {s("padding", 9, 23)}'
        elif op == 'ECB_CMD_WD_FENCE_AND_EXE':
            return f'{s("cmd_type", 0, 4)} ({"ECB_CMD_WD_FENCE_AND_EXE":24}), {s("yield", 4,1)}, {s("dma_completion", 5, 3)}, {s("reserved", 8, 19)}, {s("wd_ctxt_id", 27, 3)}, {s("reserved", 30, 2)}'
        elif op == 'ECB_CMD_SCHED_DMA':
            return f'{s("cmd_type", 0, 4)} ({"ECB_CMD_SCHED_DMA":24}), {s("yield", 4,1)}, {s("addr_index", 5, 3)}, {s("size", 8, 10)}, {s("gc_ctxt_offset", 18, 14)}'
        elif op == 'ECB_CMD_STATIC_DESC_V2':
            return f'{s("cmd_type", 0, 4)} ({"ECB_CMD_STATIC_DESC_V2":24}), {s("yield", 4,1)}, {s("reserved", 5, 3)}, {s("cpu_index", 8, 8)}, {s("size", 16, 13)}, {s("addr_index", 29, 3)}'


class Gaudi3HalReader(HalReader):
    def __init__(self):
        super().__init__('Gaudi3Graph')

    def fmt_qman_cmd_fields(self, cmd):
        s = partial(Utils.cmd_field_to_str, cmd = cmd)
        op = self.qman_cmd_to_op_str(cmd)
        if op == 'WREG_32':
            return f'{s("MB",63,1)}, {s("SWTC",62,1)}, {s("EB",61,1)}, {s("OP",56,5)} ({"WREG 32":<15}), {s("REG_OFFSET",40,16)}, {s("RSV",38,2)}, {s("PRED",32,5)}, {s("REG_ID" if Utils.ad(cmd,37,1) == 1 else "VALUE",0,32)}'
        elif op == 'WREG_BULK':
            return f'{s("MB",63,1)}, {s("SWTC",62,1)}, {s("EB",61,1)}, {s("OP",56,5)} ({"WREG BULK":<15}), {s("REG_OFFSET",40,16)}, {s("PRED",32,5)}, {s("SIZE64",0,16)}'
        elif op == 'FENCE':
            return f'{s("MB",63,1)}, {s("SWTC",62,1)}, {s("EB",61,1)}, {s("OP",56,5)} ({"FENCE":<15}), {s("PRED",32,5)}, {s("ID",30,2)}, {s("TARGET_VAL",16,14)}, {s("DEC_VAL",0,4)}'
        elif op == 'NOP':
            return f'{s("MB",63,1)}, {s("SWTC",62,1)}, {s("EB",61,1)}, {s("OP",56,5)} ({"NOP":<15})'
        elif op == 'WREG_64_SHORT':
            return f'{s("MB",63,1)}, {s("SWTC",62,1)}, {s("EB",61,1)}, {s("OP",56,5)} ({"WREG_64_SHORT":<15}), {s("DREG_OFFSET",42,14)}, {s("BASE",37,5)}, {s("PRED",32,5)}, {s("OFFSET",0,32)}'
        elif op == 'WREG_64_LONG':
            return f'{s("MB",63,1)}, {s("SWTC",62,1)}, {s("EB",61,1)}, {s("OP",56,5)} ({"WREG_64_LONG":<15}), {s("DREG_OFFSET",42,14)}, {s("REL",2,1)}, {s("BASE",37,5)}, {s("PRED",32,5)}, {s("DW_ENABLE",0,2)}'

    def fmt_ecb_cmd_fields(self, cmd):
        s = partial(Utils.cmd_field_to_str, cmd = cmd)
        op = self.ecb_cmd_to_op_str(cmd)
        if op == 'ECB_CMD_LIST_SIZE':
            return f'{s("cmd_type", 0, 4)} ({"ECB_CMD_LIST_SIZE":24}), {s("yield", 4,1)}, {s("topology_start", 5, 1)}, {s("reserved", 6, 2)}, {s("list_size", 8, 24)}'
        elif op == 'ECB_CMD_NOP':
            return f'{s("cmd_type", 0, 4)} ({"ECB_CMD_NOP":24}), {s("yield", 4,1)}, {s("dma_completion", 5, 3)}, {s("switch_cq", 8, 1)}, {s("padding", 9, 23)}'
        elif op == 'ECB_CMD_WD_FENCE_AND_EXE':
            return f'{s("cmd_type", 0, 4)} ({"ECB_CMD_WD_FENCE_AND_EXE":24}), {s("yield", 4,1)}, {s("dma_completion", 5, 3)}, {s("reserved", 8, 19)}, {s("wd_ctxt_id", 27, 3)}, {s("reserved", 30, 2)}'
        elif op == 'ECB_CMD_SCHED_DMA':
            return f'{s("cmd_type", 0, 4)} ({"ECB_CMD_SCHED_DMA":24}), {s("yield", 4,1)}, {s("addr_index", 5, 3)}, {s("size", 8, 10)}, {s("wd_type", 18, 1)}, {s("gc_ctxt_offset", 19, 13)}'
        elif op == 'ECB_CMD_STATIC_DESC_V2':
            return f'{s("cmd_type", 0, 4)} ({"ECB_CMD_STATIC_DESC_V2":24}), {s("yield", 4,1)}, {s("reserved", 5, 3)}, {s("cpu_index", 8, 8)}, {s("size", 16, 13)}, {s("addr_index", 29, 3)}'

class EagerRecipeDump(gdb.Command):
    '''Print recipe to screen for comparison based debug.
    usage: eager_recipe_dump <recipe_t*> [dump_file]'''

    def __init__(self):
        super(EagerRecipeDump, self).__init__("eager_recipe_dump", gdb.COMMAND_DATA)
        self.full_reg_map = defaultdict(list)
        self.dont_repeat()

    def __print_all_reg_assignment(self):
        print('full wreg32 and wreg_bulk sorted assignments:'),
        print('\n'.join('0x{:x} = {}'.format(k, list(map(hex, vs))) for k, vs in sorted(self.full_reg_map.items())))

    def __print_recipe_t(self):
        '''preety-print recipe_t'''

        def print_member(member_name):
            member = self.recipe[member_name]
            print('{:<24} {} = {}'.format(str(member.type), member_name, member))

        def print_with_indent(lines_str, indent=1, index=False):
            indent_str ='\t' * indent
            lines = (line.strip() for line in lines_str.split('\n') if line.strip())
            #if index:
            #    print('\n'.join(f'#{line_num:>3}: {indent_str}{line}' for line_num, line in enumerate(lines)))
            #else:
            print('\n'.join(f'{indent_str}{line}' for line in lines))
        print_member('version_major')
        print_member('version_minor')
        print()

        # TODO: infer unit from sizeof
        # TODO: good place to assert that blob pts are in range
        def print_blob_buffer(name : str, unit : int):
            size_str = name + '_blobs_buffer_size'
            ptr_str = name + '_blobs_buffer'

            print_member(size_str)
            print_member(ptr_str)
            print()

            size_val = self.recipe[size_str]
            ptr_val = self.recipe[ptr_str]
            assert size_val % unit == 0, f'expected {size_str} to be % unit == 0'

            assert unit in (8, 4)
            fmt = 'g' if unit == 8 else 'w'
            print_with_indent(gdb.execute(f'x/{size_val/unit}{fmt}x {ptr_val}', to_string=True))
            print('')

        def print_ecb(ecb):
            assert (ecb['cmds_size'] % 256) == 0, 'got an ecb with size not divided by 256!'
            ecb_cmds_hack = Utils.as_void_ptr(ecb['cmds'])
            print_with_indent(gdb.execute(f"x/{ecb['cmds_size']/4}wx {ecb_cmds_hack}", to_string=True), indent=2)

        print_member('nameSize')
        print_member('name')
        recipe_name_size = self.recipe['nameSize']
        recipe_name = self.recipe['name']
        if recipe_name_size != 0 and recipe_name != 0:
            print(f'\t{recipe_name}')
        else:
            print()
            assert recipe_name_size == 0 and recipe_name == 0, 'mismatch recipe_t::nameSize and recipe_t::name'
        print()
        print_blob_buffer('execution', 8)
        print_blob_buffer('patching', 8)
        print_blob_buffer('dynamic', 4)
        print()
        print_member('blobs_nr')
        print_member('blobs')
        print()
        recipe_blobs_nr = self.recipe['blobs_nr']
        recipe_blobs = self.recipe['blobs']
        for i in range(recipe_blobs_nr):
            print(f'\t#{i:>03}: {recipe_blobs[i]}')
        print()
        print_member('programs_nr')
        print_member('programs')
        print()
        for i in range(self.recipe['programs_nr']):
            program = self.recipe['programs'][i]
            print(f'\t#{i:>03}: {program}')
            print_with_indent(gdb.execute(f"x/{program['program_length']}gx {program['blob_indices']}", to_string=True), indent=2)
        print()
        print_member('activate_jobs_nr')
        print_member('activate_jobs')
        print()
        for i in range(self.recipe['activate_jobs_nr']):
            activate_job = self.recipe['activate_jobs'][i]
            print(f'\t{i:>03}: {activate_job}')
        print()
        print_member('arc_jobs_nr')
        print_member('arc_jobs')
        print()
        for i in range(self.recipe['arc_jobs_nr']):
            arc_job = self.recipe['arc_jobs'][i]
            print(f'\t#{i:>03}: {arc_job}')
            print()
            print('\t\tstatic_ecb:')
            print_ecb(arc_job['static_ecb'])
            print()
            print('\t\tdynamic_ecb:')
            print_ecb(arc_job['dynamic_ecb'])
            print()
        print_member('persist_tensors_nr')
        print_member('tensors')
        print()
        for i in range(self.recipe['persist_tensors_nr']):
            tensor = self.recipe['tensors'][i]
            print(f'\t#{i:>03}: {tensor}')
        print()
        print_member('program_data_blobs_size')
        print_member('program_data_blobs_buffer')
        print()
        #TODO: is this really in bytes??
        if False:
            print('WARN: not sure if this is really in bytes...')
            print_with_indent(gdb.execute(f"x/{self.recipe['program_data_blobs_size']}bx {Utils.as_void_ptr(self.recipe['program_data_blobs_buffer'])}", to_string=True))
        else:
            ptr = Utils.as_unsigned_char_ptr(self.recipe['program_data_blobs_buffer'])
            program_data_blobs_buffer = bytearray(ptr[i] for i in range(self.recipe['program_data_blobs_size']))
            print ('program_data_blobs_buffer sha1:', hashlib.sha1(program_data_blobs_buffer).hexdigest())
            print()
        print_member('program_data_blobs_nr')
        print_member('program_data_blobs')
        print()
        for i in range(self.recipe['program_data_blobs_nr']):
            program_data_blob = self.recipe['program_data_blobs'][i]
            print(f'\t{i:>03}: {program_data_blob}')
        print()
        print_member('patch_points_nr')
        print_member('patch_points')
        print()
        for i in range(self.recipe['patch_points_nr']):
            print(f"\t{i:>03}: {self.recipe['patch_points'][i]}")
        print()
        print_member('sections_nr')
        print()
        print_member('section_groups_nr')
        print_member('section_groups_patch_points')
        print()
        for i in range(self.recipe['section_groups_nr']):
            section_group = self.recipe['section_groups_patch_points'][i]
            print(f'\t#{i:>03}: {section_group}')
            print_with_indent(gdb.execute(f"x/{section_group['patch_points_nr']}wx {section_group['patch_points_index_list']}", to_string=True), indent=2)
        print()
        print_member('sobj_section_group_patch_points')
        print()
        section_group = self.recipe['sobj_section_group_patch_points']
        print_with_indent(gdb.execute(f"x/{section_group['patch_points_nr']}wx {section_group['patch_points_index_list']}", to_string=True), indent=1)
        print()
        print_member('section_ids_nr')
        print_member('section_blobs_indices')
        print()
        for i in range(self.recipe['section_ids_nr']):
            section_blob_indices = self.recipe['section_blobs_indices'][i]
            print(f'\t#{i:>03}: {section_blob_indices}')
            print_with_indent(gdb.execute(f"x/{section_blob_indices['blobs_nr']}wx {section_blob_indices['blob_indices']}", to_string=True), indent=2)
        print()
        print_member('node_nr')
        print_member('node_exe_list')
        print()
        for i in range(self.recipe['node_nr']):
            node = self.recipe['node_exe_list'][i]
            print(f'#{i:>03}: {node}')
            print_with_indent(gdb.execute(f"x/{self.recipe['programs_nr']}wx {node['program_blobs_nr']}", to_string=True), indent=2)
        print()
        print_member('workspace_nr')
        print_member('workspace_sizes')
        print()
        print_with_indent(gdb.execute(f"x/{self.recipe['workspace_nr']}gx {self.recipe['workspace_sizes']}", to_string=True), indent=1)
        print()
        print_member('debug_profiler_info')
        for i in range(self.recipe['debug_profiler_info']['num_nodes']):
            node = self.recipe['debug_profiler_info']['nodes'][i]
            print(f'#{i:>03}: {node}')
            if node['num_rois']:
                num_working_engines_hack = Utils.as_void_ptr(node['num_working_engines'])
                print_with_indent(gdb.execute(f"x/{node['num_rois']}b {num_working_engines_hack}", to_string=True), indent=2)
        if self.recipe['debug_profiler_info']['printf_addr_nr']:
            print_with_indent(gdb.execute(f"x/{self.recipe['debug_profiler_info']['printf_addr_nr']}gx {self.recipe['debug_profiler_info']['printf_addr']}"))
        print()
        print_member('recipe_conf_nr')
        print_member('recipe_conf_params')
        print()
        for i in range(self.recipe['recipe_conf_nr']):
            tmp = self.recipe['recipe_conf_params'][i]
            print(f'\t#{i:>03}: {tmp}')
        print()
        print_member('nop_kernel_offset')
        print_member('nop_kernel_section')
        print_member('valid_nop_kernel')

        print_member('max_used_mcid_discard')
        print_member('max_used_mcid_degrade')

    def __print_static_patching_cmds(self):
        # TODO: assert blob offset + size vs blob_map for static + patching blobs

        recipe_blobs_nr, recipe_blobs = self.recipe['blobs_nr'], self.recipe['blobs']
        execution_blobs_buffer_size, execution_blobs_buffer = self.recipe['execution_blobs_buffer_size'], self.recipe['execution_blobs_buffer']
        patching_blobs_buffer_size, patching_blobs_buffer = self.recipe['patching_blobs_buffer_size'], self.recipe['patching_blobs_buffer']
        dynamic_blobs_buffer_size, dynamic_blobs_buffer = self.recipe['dynamic_blobs_buffer_size'], self.recipe['dynamic_blobs_buffer']

        def static_patching_blob(base_addr, max_offset, recipe_blob_data, recipe_blob_size):
            reg_map = defaultdict(list)

            i, i_start, i_max, i_limit = 0, recipe_blob_data - base_addr, (recipe_blob_size / 8), max_offset / 8
            while i < i_max:
                assert i + i_start < i_limit, f'i={i}, i_start={i_start}, i_max={i_max}, i_limit={i_limit}'
                cmd = int(base_addr[i + i_start])
                op_str = self.hal_reader.qman_cmd_to_op_str(cmd)
                op_code = self.hal_reader.qman_cmd_to_op_code(cmd)

                print(f'#{i:>3} {cmd:#0{18}x}\t {self.hal_reader.fmt_qman_cmd_fields(cmd)}')
                if op_str == 'NOP':
                    i += 1
                elif op_str == 'WREG_32': #print(f'#{i:>3} {op_code:>5}: {"WREG 32":<15}: {ds[i]}\n\t {self.hal_reader.fmt_qman_cmd_fields(cmd)}')
                    if self.hal_reader.get_wreg_32_reg(cmd) == 0:
                        reg_offset, value = self.hal_reader.get_wreg_32_offset_and_value(cmd)
                        reg_map[reg_offset] += [value]
                    i += 1
                elif op_str == 'WREG_BULK': #print(f'#{i:>3} {op_code:>5}: {"WREG BULK":<15}: {ds[i]}\n\t {self.hal_reader.fmt_qman_cmd_fields(cmd)}')
                    bulk_size = self.hal_reader.get_wreg_bulk_size(cmd)
                    bulk_offset = self.hal_reader.get_wreg_bulk_offset(cmd)
                    for ii in range((i + 1) + i_start, (i + 1 + bulk_size) + i_start, 4):
                        line_vals = [base_addr[idx] for idx in range(ii, min(ii + 4, (i + 1 + bulk_size) + i_start))]
                        print('  ' + '  '.join([f'{int(v):#0{18}x}' for v in line_vals]))
                    for j in range(bulk_size):
                        reg_map[bulk_offset + j * 8] += [base_addr[(i + i_start) + (1 + j)] & 0xffffffff]
                        reg_map[bulk_offset + j * 8 + 4] += [(base_addr[(i + i_start) + (1 + j)] >> 32) & 0xffffffff]
                    i += bulk_size + 1
                elif op_str == 'FENCE': #print(f'#{i:>3} {op_code:>5}: {"FENCE":<15}: {ds[i]}\n\t {self.hal_reader.fmt_qman_cmd_fields(cmd)}')
                    i += 1
                elif op_str == 'WREG_64_SHORT': #print(f'#{i:>3} {op_code:>5}: {"WREG_64_SHORT":<15}: {ds[i]}\n\t {self.hal_reader.fmt_qman_cmd_fields(cmd)}')
                    i += 1
                elif op_str == 'WREG_64_LONG': #print(f'#{i:>3} {op_code:>5}: {"WREG_64_LONG":<15}: {ds[i]}\n\t {self.hal_reader.fmt_qman_cmd_fields(cmd)}')
                    print(f'  {int(base_addr[(i + 1) + i_start]):#0{18}x}')
                    i += 1 + 1
                else:
                    assert False, f'unknown opcode {op_code} at idx {i}'
                    i += 1
            # if True:
            #     if blob is not None:
            #         assert i == recipe_blob_size / 8, f'i={i}, size={recipe_blob_size / 8} mismatch!'
            print()
            print('wreg32 and wreg_bulk assignments:')
            print(', '.join('0x{:x} = {}'.format(k, list(map(hex, vs))) for k, vs in reg_map.items()))
            print()
            print('wreg32 and wreg_bulk sorted assignments:')
            print('\n'.join('0x{:x} = {}'.format(k, list(map(hex, vs))) for k, vs in sorted(reg_map.items())))
            print()

            self.full_reg_map.update(reg_map)

        def __print_dynamic_blob(offset, size, engine):
            ENGINE_TO_WD = {
                'Recipe::TPC' : 'tpc_wd_ctxt_t',
                'Recipe::MME' : 'mme_wd_ctxt_t',
                'Recipe::DMA' : 'edma_wd_ctxt_t',
                'Recipe::ROT' : 'rot_wd_ctxt_t'
            }
            assert engine in ENGINE_TO_WD, f'unkown engine {engine}'
            wd_type = self.hal_reader.get_wd_type(ENGINE_TO_WD[engine])
            assert size == wd_type.sizeof, f'unexpected work distribution context size {size}'
            print(f'offset: {offset:>5} type: {engine}\t|\t', str((Utils.as_unsigned_char_ptr(dynamic_blobs_buffer) + offset).cast(wd_type.pointer()).dereference()))
            print()

        def __dynamic_blob(blob_offset):
            for offset, size, engine in self.blob_map['dynamic']:
                if offset == blob_offset:
                    __print_dynamic_blob(offset, size, engine)
                    break
            else:
                assert False, f'No dynamic blob at base_addr {blob_offset} found! known dynamic blobs: {self.blob_map["dynamic"]}'

        for blob_idx in range(recipe_blobs_nr):
            print('''
/===========================================================\\
|                                                           |
|                       Blob #{:>06}                        |
|                                                           |
\\===========================================================/
'''.format(blob_idx))

            blob = recipe_blobs[blob_idx]

            print('{:<24} [{}] = {}'.format(str(blob.type), blob_idx, blob))
            print()

            if blob['blob_type']['requires_patching']:
                base_addr = patching_blobs_buffer     # uint64_t*
                max_offset = patching_blobs_buffer_size # in bytes
                recipe_blob_data = Utils.as_uint64_t_ptr(blob['data'])
            elif blob['blob_type']['static_exe']:
                base_addr = execution_blobs_buffer    # uint64_t*
                max_offset = execution_blobs_buffer_size # in bytes
                recipe_blob_data = Utils.as_uint64_t_ptr(blob['data'])
            else:
                base_addr = dynamic_blobs_buffer    # uint32_t*
                max_offset = dynamic_blobs_buffer_size # in bytes
                recipe_blob_data = Utils.as_uint32_t_ptr(blob['data'])

            recipe_blob_size = blob['size'] # size in bytes
            assert Utils.as_unsigned_char_ptr(recipe_blob_data) + recipe_blob_size <= Utils.as_unsigned_char_ptr(base_addr) + max_offset, "blob out of range?!"

            if blob['blob_type']['requires_patching'] or blob['blob_type']['static_exe']:
                assert recipe_blob_size % 8 == 0, 'expected to be working with byte sizes of uint64_t array'
                assert max_offset % 8 == 0, 'expected to be working with byte sizes of uint64_t array'
                static_patching_blob(base_addr, max_offset, recipe_blob_data, recipe_blob_size)
            else:
                assert recipe_blob_size % 4 == 0, 'expected to be working with byte sizes of uint32_t array'
                assert max_offset % 4 == 0, 'expected to be working with byte sizes of uint32_t array'
                __dynamic_blob(int(blob['data']) - int(base_addr))
        
        print(R'''
/===========================================================\
|                                                           |
|           (dynamic blobs summary (repeated data))         |
|                                                           |
\===========================================================/
''')
        for offset, size, engine in sorted(self.blob_map['dynamic']):
            __print_dynamic_blob(offset, size, engine)

    def __print_ecb_cmds(self):
        ENGINE_TYPES = ['Recipe::TPC', 'Recipe::MME', 'Recipe::DMA', 'Recipe::ROT']

        arc_jobs_nr = self.recipe['arc_jobs_nr']
        arc_jobs = self.recipe['arc_jobs']
        if not arc_jobs_nr:
            print('No Arc Jobs ???')
            print()
            return
        
        arc_engines = [str(arc_jobs[i]['logical_engine_id']) for i in range(arc_jobs_nr)]
        assert len(set(arc_engines)) == len(arc_engines), f'repeated values in arc engines: {arc_engines}'
        assert set(arc_engines).issubset(ENGINE_TYPES), f'unknown engine in arc engines: {arc_engines} allowed values: {ENGINE_TYPES}'

        blob_map = {'static': set(), 'dynamic': set(), 'patching': set()}

        # TODO: maybe sort the arc jobs by engine type otherwise it'll be harder to diff them
        for arc_job_idx in range(arc_jobs_nr):
            arc_job = arc_jobs[arc_job_idx]
            engine_str = str(arc_job['logical_engine_id'])

            print('''
/===========================================================\\
|                                                           |
|           ARC JOB #{} Engine Type: {}             |
|                                                           |
\\===========================================================/
'''.format(arc_job_idx, engine_str))

            print('{:<24} [{}] = {}'.format(str(arc_job.type), arc_job_idx, arc_job))
            print()

            for sd in ['static_ecb', 'dynamic_ecb']:
                ecbs = arc_job[sd]
                cmds_size = ecbs['cmds_size']
                cmds_eng_offset = ecbs['cmds_eng_offset']

                assert cmds_eng_offset % 256 == 0
                assert cmds_eng_offset == 0 or cmds_size % cmds_eng_offset == 0

                num_eng = 1 if cmds_eng_offset == 0 else cmds_size / cmds_eng_offset
                for ecb_idx in range(num_eng):
                    print(f'\t{engine_str} - {sd} - ECB #{ecb_idx} (Total: {num_eng})')
                    print()

                    # all stuff is in bytes here...
                    # base_addr = as_unsigned_char_ptr(self.recipe['execution_blobs_buffer'] if sd == 'static_ecb' else self.recipe['dynamic_blobs_buffer'])
                    # i_limit = self.recipe['execution_blobs_buffer_size'] if sd == 'static_ecb' else self.recipe['dynamic_blobs_buffer_size']
                    # i_start = as_unsigned_char_ptr(ecbs['cmds']) - base_addr

                    base_addr = ecbs['cmds']

                    i = int(cmds_eng_offset) * ecb_idx                    
                    i_max = cmds_size if num_eng == 1 else cmds_eng_offset * (ecb_idx + 1)
                    while i < i_max:

                        assert (i) % 4 == 0
                        cmd = int(Utils.as_uint32_ptr(base_addr)[i / 4])
                        op_str = self.hal_reader.ecb_cmd_to_op_str(cmd)
                        print(f'byte offset: {i:>5}\t|\t', end ='')
                        if op_str == 'ECB_CMD_LIST_SIZE':
                            print(self.hal_reader.fmt_ecb_cmd_fields(cmd))
                            i += 4
                        elif op_str == 'ECB_CMD_NOP':
                            print(self.hal_reader.fmt_ecb_cmd_fields(cmd))
                            i += (1 + self.hal_reader.get_ecb_nop_padding(cmd)) * 4
                        elif op_str == 'ECB_CMD_WD_FENCE_AND_EXE':
                            assert sd == 'dynamic_ecb'
                            print(self.hal_reader.fmt_ecb_cmd_fields(cmd))
                            i += 4
                        elif op_str == 'ECB_CMD_SCHED_DMA':
                            assert sd == 'dynamic_ecb'
                            addr_index = self.hal_reader.get_ecb_sched_dma_addr_index(cmd)
                            addr_offset = int(Utils.as_uint32_ptr(base_addr)[i / 4 + 1])
                            copy_buffer_size = self.hal_reader.get_ecb_sched_dma_size(cmd)
                            
                            assert addr_index == 2, f'expected addr_index in ECB_CMD_SCHED_DMA to be 2 (dynamic) but got {addr_index}'
                            buffer_size = self.recipe['dynamic_blobs_buffer_size']
                            blob_map['dynamic'].add((addr_offset, copy_buffer_size, engine_str))
                            print(self.hal_reader.fmt_ecb_cmd_fields(cmd), end='')
                            print (', addr_offset: ', int(Utils.as_uint32_ptr(base_addr)[i / 4 + 1]))
                            i += 8
                        elif op_str == 'ECB_CMD_STATIC_DESC_V2':
                            assert sd == 'static_ecb', 'ECB_CMD_STATIC_DESC_V2 in non static ecb ??'
                            addr_index = self.hal_reader.get_ecb_static_desc_v2_addr_index(cmd)
                            addr_offset = int(Utils.as_uint32_ptr(base_addr)[i / 4 + 1])
                            copy_buffer_size = self.hal_reader.get_ecb_static_desc_v2_size(cmd)

                            if addr_index == 0: # patching
                                buffer_size = self.recipe['patching_blobs_buffer_size']
                                blob_map['patching'].add((addr_offset, copy_buffer_size, engine_str))
                            elif addr_index == 1: # execute
                                buffer_size = self.recipe['execution_blobs_buffer_size']
                                blob_map['static'].add((addr_offset, copy_buffer_size, engine_str))
                            else:
                                assert False, f'expected addr_index in ECB_CMD_STATIC_DESC_V2 to be either 0 (patching) or 1 (execute) but got {addr_index}'
                            assert addr_offset < buffer_size, 'expected {} < {}'.format(addr_offset, int(buffer_size))
                            assert addr_offset + copy_buffer_size <= buffer_size, 'expected {} <= {}'.format(addr_offset + copy_buffer_size, int(buffer_size))

                            print(self.hal_reader.fmt_ecb_cmd_fields(cmd), end='')
                            print (', addr_offset: ', addr_offset)
                            i += 8
                        else:
                            assert False, 'Unsupported ECB op'
                    assert i == i_max
                    print()
        
        # add sanity checks over blob_map -> expecting no segment overlap and segments completely covering the whole size without gaps
        def blob_map_sanity(blob_set, max_size, label):
            data = sorted(set((a, b) for (a, b, _) in blob_set))
            assert data[0][0] == 0, f'{label}: data[0][0] is {data[0][0]} whereas expected to start reading dynamic blobs from offset 0'
            assert all(a + b == c for ((a, b), (c, _)) in zip(data, data[1:])), f'{label}: non perfect coverage upto max_size {max_size} with data: {data}'
            assert sum(data[-1][:2]) == max_size, f'{label}: last data start and len are {data[-1][:2]} which do not sum to the max_size, {max_size}'

        assert len(blob_map) == 3, "i've changed blob_map and haven't updated the code here???"
        blob_map_sanity(blob_map['dynamic'], self.recipe['dynamic_blobs_buffer_size'], 'dynamic')
        blob_map_sanity(blob_map['patching'], self.recipe['patching_blobs_buffer_size'], 'patching')
        blob_map_sanity(blob_map['static'], self.recipe['execution_blobs_buffer_size'], 'static')

        self.blob_map = blob_map

    @staticmethod
    def usage():
        print('usage: eager_recipe_dump <recipe_t*> [dump_file]')

    def __set_device_type(self):
        for i in range(self.recipe['recipe_conf_nr']):
            conf = self.recipe['recipe_conf_params'][i]
            if (str(conf['conf_id']) == 'gc_conf_t::DEVICE_TYPE'):
                device_types = gdb.lookup_type('synDeviceType')
                if device_types['synDeviceGaudi2'].enumval == conf['conf_value']:
                    self.device_type = 'Gaudi2'
                    break
                elif device_types['synDeviceGaudi3'].enumval == conf['conf_value']:
                    self.device_type = 'Gaudi3'
                    break
                else:
                    assert False, f'got an unsupported device type {conf["conf_value"]}'
        else:
            assert False, 'configuration params are missing device type'

    def __set_hal_Reader(self):
        if self.device_type == 'Gaudi2':
            self.hal_reader = Gaudi2HalReader()
        elif self.device_type == 'Gaudi3':
            self.hal_reader =  Gaudi3HalReader()
        else:
            assert False, f'Unsupported device type for HalReader {self.device_type}'

    def invoke(self, arg: str, from_tty: bool):
        args = arg.split()
        if not len(args) in (1, 2):
            print(f'expected 1 or 2 args but got {len(args)}')
            self.usage()
            return

        recipe_arg = args[0]
        # dump_file_arg = args[1] if len(args) >= 2 else None

        #recipe_ptr = gdb.parse_and_eval(recipe_arg)

        self.recipe = gdb.parse_and_eval(f'*{recipe_arg}')
        if self.recipe.type.name != 'recipe_t':
            print('1st argument should be a ptr to "recipt_t"')
            self.usage()
            return

        self.__set_device_type()
        self.__set_hal_Reader()

        print(R'''
/===========================================================================\
|                                                                           |
|                                                                           |
|       /===========================================================\       |
|       |                                                           |       |
|       |                     recipe_t content                      |       |
|       |                                                           |       |
|       \===========================================================/       |
|                                                                           |
|                                                                           |
\===========================================================================/
''')
        self.__print_recipe_t()
        print()

        print(R'''
/===========================================================================\
|                                                                           |
|                                                                           |
|       /===========================================================\       |
|       |                                                           |       |
|       |               ecb_t command buffer decode                 |       |
|       |                                                           |       |
|       \===========================================================/       |
|                                                                           |
|                                                                           |
\===========================================================================/
''')
        self.__print_ecb_cmds()
        print()

        print(R'''
/===========================================================================\
|                                                                           |
|                                                                           |
|       /===========================================================\       |
|       |                                                           |       |
|       |           static + patching command buffer decode         |       |
|       |                                                           |       |
|       \===========================================================/       |
|                                                                           |
|                                                                           |
\===========================================================================/
''')
        self.__print_static_patching_cmds()
        print()

        self.__print_all_reg_assignment()
        print()

EagerRecipeDump()