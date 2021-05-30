<?php

	//require_once "../../src/models/forcastingService/timeSerie.php";
	require_once "../../src/controllers/forcastingService/TSCManager.php";
	
	session_start();
	$tsc = isset($_SESSION['tsc']) ? $_SESSION['tsc'] : NULL;
	//var_dump($tsc);
	$totalNan = $tsc->getNanValuesCount();
	if($totalNan > 0){
		header("Location: showTimeSerie.php");
	}
	$indArr = TSC_Manager::getIndexArray($tsc);
	//var_dump($indArr);


?>
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>Smart SV prediction | selection parametres prediction</title>
	<link rel="stylesheet" type="text/css" href="../../css/bootstrap/css/bootstrap.min.css">
	<script src="../../js/predictParam.js"></script>
</head>
<body>
<form action="../../src/quit.php" method="post"><input class="btn btn-danger" name="quitButton" value="Quit" type="submit"></form>
	<form action="../../src/doPrediction.php" method="POST">
		<table style="width: 80%;">
			<tr>
				<td>
					<strong>Colonnes X</strong> <br>
					<?php
					for($i=0;$i<count($indArr);$i++){
						echo "<label for='".$indArr[$i]."'>".$indArr[$i]."</label>";
						echo "<input type='checkbox' id='".$indArr[$i]."' name='xCols[]' value='".$indArr[$i]."' onchange=\"checkboxChecking(document.getElementById('".$indArr[$i]."'));\"><br>";
					}
					?>
				</td>
				<td>
					<label for="yCol">Colonne Y selectionnées</label><br>
					<select class="form-select" name="yCol">
					<?php
						for($i=0;$i<count($indArr);$i++){
							echo "<option value=\"".$indArr[$i]."\">".$indArr[$i]."</option>";
						}
					?>
					</select>
				</td>
			</tr>
			<tr>
				<td>
					<label for="svr-poly">SVR_poly</label>
					<input type="checkbox" id="svr-poly" name="method[]" value="SVR_poly" onchange="checkboxChecking(document.getElementById('svr-poly'));"><br>

					<label for="svr-rbf">SVR_rbf</label>
					<input type="checkbox" id="svr-rbf" name="method[]" value="SVR_rbf" onchange="checkboxChecking(document.getElementById('svr-rbf'));"><br>

					<label for="svr-linear">SVR_linear</label>
					<input type="checkbox" id="svr-linear" name="method[]" value="SVR_linear" onchange="checkboxChecking(document.getElementById('svr-linear'));"><br><br>

					<label for="sgd-regressor">SGDRegressor</label>
					<input type="checkbox" id="sgd-regressor" name="method[]" value="SGDRegressor" onchange="checkboxChecking(document.getElementById('sgd-regressor'));"><br>

					<label for="bayesian-ridge">BayesianRidge</label>
					<input type="checkbox" id="bayesian-ridge" name="method[]" value="BayesianRidge" onchange="checkboxChecking(document.getElementById('bayesian-ridge'));"><br>

					<label for="theilsen-regressor">TheilSenRegressor</label>
					<input type="checkbox" id="theilsen-regressor" name="method[]" value="TheilSenRegressor" onchange="checkboxChecking(document.getElementById('theilsen-regressor'));"><br><br>

					<label for="linear-regression">LinearRegression</label>
					<input type="checkbox" id="linear-regression" name="method[]" value="LinearRegression" onload="defaultMethods(document.getElementById('linear-regression'));" onchange="checkboxChecking(document.getElementById('linear-regression'));"><br>

					<label for="huber-regressor">HuberRegressor</label>
					<input type="checkbox" id="huber-regressor" name="method[]" value="HuberRegressor" onload="defaultMethods(document.getElementById('huber-regressor'));" onchange="checkboxChecking(document.getElementById('huber-regressor'));"><br><br>

					<label for="mlp-regressor2">MLPRegressor2</label>
					<input type="checkbox" id="mlp-regressor2" name="method[]" value="MLPRegressor2" onchange="checkboxChecking(document.getElementById('mlp-regressor2'));"><br>

					<label for="mlp-regressor3">MLPRegressor3</label>
					<input type="checkbox" id="mlp-regressor3" name="method[]" value="MLPRegressor3" onchange="checkboxChecking(document.getElementById('mlp-regressor3'));"><br><br>

					<label for="knn-regressor">KNNRegressor</label>
					<input type="checkbox" id="knn-regressor" name="method[]" value="KNNRegressor" onchange="checkboxChecking(document.getElementById('knn-regressor'));"><br>

					<label for="lasso-lars">LassoLars</label>
					<input type="checkbox" id="lasso-lars" name="method[]" value="LassoLars" onchange="checkboxChecking(document.getElementById('lasso-lars'));"><br>

					<label for="lars">Lars</label>
					<input type="checkbox" id="lars" name="method[]" value="Lars" onchange="checkboxChecking(document.getElementById('lars'));"><br><br>

					<label for="elastic-net-regressor">ElasticNetRegressor</label>
					<input type="checkbox" id="elastic-net-regressor" name="method[]" value="ElasticNetRegressor" onchange="checkboxChecking(document.getElementById('elastic-net-regressor'));"><br>

					<label for="passive-agressive-regressor">PassiveAgressiveRegressor</label>
					<input type="checkbox" id="passive-agressive-regressor" name="method[]" value="PassiveAggressiveRegressor" onchange="checkboxChecking(document.getElementById('passive-agressive-regressor'));"><br>

					<label for="ransac-regressor">RANSACRegressor</label>
					<input type="checkbox" id="ransac-regressor" name="method[]" value="RANSACRegressor" onchange="checkboxChecking(document.getElementById('ransac-regressor'));"><br>

					<label for="orthogonal-matching-pursuit">OrthogonalMatchingPursuit</label>
					<input type="checkbox" id="orthogonal-matching-pursuit" name="method[]" value="OrthogonalMatchingPursuit" onchange="checkboxChecking(document.getElementById('orthogonal-matching-pursuit'));"><br>

					<label for="extra-trees-regressor">ExtraTreesRegressor</label>
					<input type="checkbox" id="extra-trees-regressor" name="method[]" value="ExtraTreesRegressor" onchange="checkboxChecking(document.getElementById('extra-trees-regressor'));"><br>

					<label for="gradient-boosting-regressor">GradientBoostingRegressor</label>
					<input type="checkbox" id="gradient-boosting-regressor" name="method[]" value="GradientBoostingRegressor" onchange="checkboxChecking(document.getElementById('gradient-boosting-regressor'));"><br>

				</td>
				<td>
					<label for="selectedMethods">Methodes selectionnées</label><br>
					<textarea class="form-control" id="selectedMethods" name="selectedMethods" rows="25" cols="40"></textarea>
				</td>
				<td>
					<script>var lstMthd = selectAll(); var lstDefaultMethod = selectDefaultMethod();</script>


					<button id="default" name="classicPredictButton" class="btn btn-success">Prediction par defaut</button><br><br>
					<label for="addAll">Ajouter toutes les methodes : </label>
					<button style="color: white;" class="btn btn-info" id="addAll" name="addAll" onclick=" addAllMethods(lstMthd);" type='button'>&gt;&gt;</button><br><br>
					<label for="removeAll">Supprimer toutes les methodes : </label>
					<button style="color: white;" class="btn btn-info" id="removeAll" name="removeAll" onclick=" removeAllMethods(lstMthd);" type='button'>&lt;&lt;</button><br><br>


					<label for="previousValueField">Nombre de valeurs précédentes :</label><br>
					<input class="form-control" type="text" id="previousValueField" name="previousValueField"><br><br>

					<label for="predictValsField">Nombre de valeurs à prédire (longueur prediction):</label><br>
					<input class="form-control" type="text" id="predictValsField" name="predictValsField"><br><br>

					<label for="percentTraining">Pourcentage d'entrainement</label><br>
					<input class="form-control" type="text" id="percentTraining" name="percentTraining"><br><br>

					<input class="btn btn-success" type="submit" name="submit" value="Faire prediction">
				</td>
			</tr>
		</table>
	</form>
	

</body>
</html>