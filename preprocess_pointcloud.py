from src.app import PCDAnalyzer


pcd_analyzer = PCDAnalyzer(_config=PCDAnalyzer.read_config())
pcd_analyzer.preprocess()
