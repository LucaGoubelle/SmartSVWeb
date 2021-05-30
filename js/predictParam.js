//dynamic management of methods selection for forcasting page
//author: Luca Goubelle

function selectAll(){
	var svr_poly = document.getElementById("svr-poly");
	var svr_rbf = document.getElementById("svr-rbf");
	var svr_linear = document.getElementById("svr-linear");
	var sgd_regressor = document.getElementById("sgd-regressor");
	var bayesian_ridge = document.getElementById("bayesian-ridge");

	var theilsen_regressor = document.getElementById("theilsen-regressor");
	var linear_regression = document.getElementById("linear-regression");
	var huber_regressor = document.getElementById("huber-regressor");
	var mlp_regressor2 = document.getElementById("mlp-regressor2");
	var mlp_regressor3 = document.getElementById("mlp-regressor3");

	var knn_regressor = document.getElementById("knn-regressor");
	var lasso_lars = document.getElementById("lasso-lars");
	var lars = document.getElementById("lars");
	var elastic_net_regressor = document.getElementById("elastic-net-regressor");
	var passive_agressive_regressor = document.getElementById("passive-agressive-regressor");

	var ransac_regressor = document.getElementById("ransac-regressor");
	var orthogonal_matching_pursuit = document.getElementById("orthogonal-matching-pursuit");
	var extra_trees_regressor = document.getElementById("extra-trees-regressor");
	var gradient_boosting_regressor = document.getElementById("gradient-boosting-regressor");

	return [
		svr_poly,
		svr_rbf,
		svr_linear,
		sgd_regressor,
		bayesian_ridge,
		theilsen_regressor,
		linear_regression,
		huber_regressor,
		mlp_regressor2,
		mlp_regressor3,
		knn_regressor,
		lasso_lars,
		lars,
		elastic_net_regressor,
		passive_agressive_regressor,
		ransac_regressor,
		orthogonal_matching_pursuit,
		extra_trees_regressor,
		gradient_boosting_regressor
	];
}

function selectDefaultMethod(){
	var linear_regression = document.getElementById("linear-regression");
	var huber_regressor = document.getElementById("huber-regressor");

	return [
		linear_regression,
		huber_regressor
	];
}

const varToString = varObj => Object.keys(varObj)[0]

function addMethodsToListBox(element){
	var box = document.getElementById("selectedMethods");
	if(element.checked==true)
		box.innerHTML += element.id+"\n";
}

function removeMethodsToListBox(element){
	var box = document.getElementById("selectedMethods");
	const elementToPop = element.id+'\n';
	if(element.checked==false)
	{
		var newContent = box.innerHTML.replace(elementToPop, '');
		box.innerHTML = newContent;
	}
}

function defaultMethods(element){
	var box = document.getElementById("selectedMethods");
	if(element.checked==true)
		box.innerHTML += element.id+"\n";
}

function checkboxChecking(element){
	addMethodsToListBox(element);
	removeMethodsToListBox(element);
}

function addAllMethods(arrMth){
	for(let i=0;i<arrMth.length;i++)
		arrMth[i].checked = true;
	for(let i=0;i<arrMth.length;i++)
		addMethodsToListBox(arrMth[i]);

}

function addDefaultMethod(arrMth){
	for(let i=0;i<arrMth.length;i++)
		arrMth[i].checked = true;
	for(let i=0;i<arrMth.length;i++)
		addMethodsToListBox(arrMth[i]);
}

function removeAllMethods(arrMth){
	for(let i=0;i<arrMth.length;i++)
		arrMth[i].checked = false;/*
	for(let i=0;i<arrMth.length;i++)
		removeMethodsToListBox(arrMth[i]);*/
	var box = document.getElementById("selectedMethods");
	box.innerHTML = "";
}