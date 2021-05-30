
function showTableBody(maxA, minA, nanA){
    //console.log(maxA);console.log(minA);console.log(nanA);
    var tb = document.getElementById("tableColumnsContent");
    var cnt = nanA.length;

    for(let i=0;i<cnt;i++){
        tb.innerHTML += "<tr><td>"+String(maxA[i])+"</td><td>"+String(minA[i])+"</td><td>"+String(nanA[i])+"</td></tr>";
    }
}

function showNotNanColumnsCheckbox(nanLst){
    var ldiv = document.getElementById("columnsNotNan");
    var cntNan = nanLst.length;

    for(let i=0;i<cntNan;i++){
        ldiv.innerHTML += "<label for=\""+nanLst[i]+"\">"+nanLst[i]+"</label><input type=\"checkbox\" id=\""+nanLst[i]+"\" name=\"column[]\" value=\"column_"+String(i+1)+"\" onchange=\"checkboxChecking(document.getElementById('"+nanLst[i]+"'));\"><br>";
    }
}

function addColumnToListBox(element){
	var box = document.getElementById("columnsChosen");
	if(element.checked==true)
		box.innerHTML += element.id+"\n";
}

function removeColumnFromListBox(element){
	var box = document.getElementById("columnsChosen");
	const elementToPop = element.id+'\n';
	if(element.checked==false)
	{
		var newContent = box.innerHTML.replace(elementToPop, '');
		box.innerHTML = newContent;
	}
}

function checkboxChecking(element){
	addColumnToListBox(element);
	removeColumnFromListBox(element);
}

function getNotNanColumnsIndexList()
{
    var textarea = document.getElementById("columnsChosen");
    var content = textarea.innerHTML.split("\n");
    return content;
}

function addAllMethods(arrMth){
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
	var box = document.getElementById("columnsChosen");
	box.innerHTML = "";
}