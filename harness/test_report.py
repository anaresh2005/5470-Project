import tempfile
import unittest
from pathlib import Path
from report import read_logs, summarize, write_report


def log(algorithms=('nccl', 'generated'), complete=True):
    lines = ['CHECK,reference_cpu,PASS']
    for i in range(3):
        for alg in algorithms:
            ms = {'nccl': 2, 'generated': 1, 'handwritten': 3}[alg]
            lines.append(f'RESULT,{alg},12288,{i},1,3,5470,{ms},0,PASS')
    if complete:
        lines.append('PASS: every sample on all four ranks matches ncclAllReduce')
    return '\n'.join(lines)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'fixture.log'

    def read(self, content):
        self.path.write_text(content)
        return read_logs([self.path])

    def test_paired_speedup_and_report(self):
        rows = self.read(log())
        result = write_report(rows, Path(self.tmp.name) / 'summary')
        gen = next(r for r in result if r['algorithm'] == 'generated')
        self.assertEqual(gen['median_paired_speedup'], 2)
        self.assertEqual(gen['samples'], 3)
        self.assertTrue((Path(self.tmp.name) / 'summary/poster_results.txt').exists())

    def test_reject_incomplete_failed_nonfinite_duplicate(self):
        for content in [log(complete=False), log().replace(',0,PASS', ',0,FAIL'),
                        log().replace(',2,0,PASS', ',nan,0,PASS'),
                        log() + '\nRESULT,nccl,12288,0,1,3,5470,2,0,PASS']:
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.read(content)

    def test_unpaired_samples_fail(self):
        rows = self.read(log().replace('RESULT,generated,12288,2,1,3,5470,1,0,PASS', ''))
        with self.assertRaises(ValueError):
            summarize(rows)

    def test_no_speedup_without_timed_reference(self):
        rows = self.read(log(('generated',)))
        self.assertEqual(summarize(rows)[0]['median_paired_speedup'], '')

    def test_no_pairing_across_files(self):
        rows = self.read(log(('generated',)))
        self.path = self.path.with_name('separate-job.log')
        rows += self.read(log(('nccl',)))
        generated = next(r for r in summarize(rows) if r['algorithm'] == 'generated')
        self.assertEqual(generated['median_paired_speedup'], '')

if __name__ == '__main__':
    unittest.main()
